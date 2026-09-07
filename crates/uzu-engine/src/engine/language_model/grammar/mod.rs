use thiserror::Error;
use tokenizers::Tokenizer;
use xgrammar::{
    DLDataType, DLDevice, DLDeviceType, DLTensor, Grammar as XGrammarGrammar, GrammarCompiler, GrammarMatcher,
    TokenizerInfo, c_void,
};

use crate::{data_type::DataType, engine::language_model::grammar::engagement::GrammarEngagementState};

mod config;
mod data_type;
mod engagement;
mod thinking_budget;

pub use config::GrammarConfig;
use thinking_budget::ThinkingBudget;

// TODO: jumpforward?

pub struct Grammar {
    vocab_size: usize,
    matcher: Option<GrammarMatcher>,
    engagement_state: GrammarEngagementState,
    thinking_budget: Option<ThinkingBudget>,
}

#[derive(Debug, Error)]
pub enum GrammarError {
    #[error("Grammar rejected the token")]
    GrammarReject,
    #[error("XGrammar error: {0}")]
    XGrammar(String),
}

impl Grammar {
    pub fn new(
        config: &GrammarConfig,
        tokenizer: &Tokenizer,
        trigger_token_sequence: Option<Vec<u64>>,
        stop_token_ids: Option<&[i32]>,
    ) -> Result<Self, GrammarError> {
        let tokenizer_info =
            TokenizerInfo::from_huggingface(tokenizer, None, stop_token_ids).map_err(GrammarError::XGrammar)?;

        let vocab_size = tokenizer_info.vocab_size();

        let grammar = match config {
            GrammarConfig::JsonSchema {
                schema,
                any_whitespace,
                indent,
                separators,
                strict_mode,
            } => {
                let separators_ref = separators.as_ref().map(|(a, b)| (a.as_str(), b.as_str()));
                XGrammarGrammar::from_json_schema(
                    schema,
                    *any_whitespace,
                    *indent,
                    separators_ref,
                    *strict_mode,
                    None,
                    false,
                    false,
                )
                .map_err(GrammarError::XGrammar)?
            },
            GrammarConfig::Regex {
                pattern,
                print_converted_ebnf,
            } => XGrammarGrammar::from_regex(pattern, *print_converted_ebnf).map_err(GrammarError::XGrammar)?,
            GrammarConfig::BuiltinJson => XGrammarGrammar::builtin_json_grammar(),
        };
        let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, true, -1).map_err(GrammarError::XGrammar)?;
        let compiled = compiler.compile_grammar(&grammar).map_err(GrammarError::XGrammar)?;
        let matcher = GrammarMatcher::new(&compiled, None, true, -1).map_err(GrammarError::XGrammar)?;

        let engagement_state = match trigger_token_sequence.filter(|sequence| !sequence.is_empty()) {
            Some(trigger_sequence) => GrammarEngagementState::Triggered {
                trigger_sequence,
                match_history: Vec::new(),
            },
            None => GrammarEngagementState::Always,
        };

        Ok(Self {
            vocab_size,
            matcher: Some(matcher),
            engagement_state,
            thinking_budget: None,
        })
    }

    pub fn thinking_budget(
        budget: usize,
        tokenizer: &Tokenizer,
        end_sequence: Vec<u64>,
        forced_sequence: Vec<u64>,
    ) -> Self {
        Self {
            vocab_size: tokenizer.get_vocab_size(true),
            matcher: None,
            engagement_state: GrammarEngagementState::Always,
            thinking_budget: Some(ThinkingBudget::new(budget, end_sequence, forced_sequence, tokenizer)),
        }
    }

    pub fn with_thinking_budget(
        mut self,
        budget: usize,
        tokenizer: &Tokenizer,
        end_sequence: Vec<u64>,
        forced_sequence: Vec<u64>,
    ) -> Self {
        self.thinking_budget = Some(ThinkingBudget::new(budget, end_sequence, forced_sequence, tokenizer));
        self
    }
}

impl Grammar {
    pub fn next_bitmask(
        &mut self,
        bitmask: &mut [u32],
    ) -> bool {
        let vocab_size_in_u32s = self.vocab_size.div_ceil(DataType::U32.size_in_bits());
        assert!(bitmask.len() >= vocab_size_in_u32s); // NOTE: tokenizer vocab can be smaller than model vocab

        let mut constrained = if self.engagement_state.is_engaged()
            && let Some(matcher) = self.matcher.as_mut()
        {
            let mut shape_i64 = [vocab_size_in_u32s as i64];
            let mut bitmask_tensor = unsafe {
                DLTensor::new(
                    bitmask.as_mut_ptr() as *mut c_void,
                    DLDevice {
                        device_type: DLDeviceType::kDLCPU,
                        device_id: 0,
                    },
                    1,
                    DLDataType {
                        code: 0,
                        bits: 32,
                        lanes: 1,
                    },
                    shape_i64.as_mut_ptr(),
                    core::ptr::null_mut(),
                    0,
                )
            };

            bitmask[vocab_size_in_u32s..].fill(0);
            matcher.fill_next_token_bitmask(&mut bitmask_tensor, 0, false)
        } else {
            bitmask.fill(u32::MAX);

            false
        };

        if let Some(token_id) = self.thinking_budget.as_mut().and_then(ThinkingBudget::next_forced_token) {
            let token_id = token_id as usize;
            assert!(token_id < self.vocab_size, "forced thinking marker token is outside the tokenizer vocabulary");
            bitmask.fill(0);
            bitmask[token_id / u32::BITS as usize] |= 1 << (token_id % u32::BITS as usize);
            constrained = true;
        }

        constrained
    }

    pub fn accept_token(
        &mut self,
        token_id: u64,
    ) -> Result<(), GrammarError> {
        // Speculators propose tokens before the target sampler applies the
        // bitmask. Reject branches that do not follow a pending forced marker.
        // Forced transition whitespace is protocol syntax rather than final
        // output, so do not feed it to a response-format matcher.
        let forced_token = self.thinking_budget.as_mut().and_then(ThinkingBudget::next_forced_token);
        if forced_token.is_some_and(|forced_token| forced_token != token_id) {
            return Err(GrammarError::GrammarReject);
        }

        // A terminated matcher cannot advance and its bitmask only allows stop
        // tokens, which close generation without being part of the grammar, so
        // tokens sampled after termination must not be rejected.
        if forced_token.is_none()
            && self.engagement_state.is_engaged()
            && let Some(matcher) = self.matcher.as_mut()
            && !matcher.is_terminated()
            && !matcher.accept_token(token_id as i32)
        {
            return Err(GrammarError::GrammarReject);
        }

        self.engagement_state.accept_token(token_id);
        if let Some(thinking_budget) = self.thinking_budget.as_mut() {
            if thinking_budget.accept_token(token_id) {
                self.engagement_state.force_engage();
            }
        }
        Ok(())
    }

    pub fn rollback(
        &mut self,
        num_tokens: usize,
    ) {
        let num_grammar_tokens = self.engagement_state.rollback(num_tokens);

        if num_grammar_tokens > 0 {
            if let Some(matcher) = self.matcher.as_mut() {
                matcher.rollback(num_grammar_tokens as i32);
            }
        }
        if let Some(thinking_budget) = self.thinking_budget.as_mut() {
            thinking_budget.rollback(num_tokens);
        }
    }

    pub fn is_terminated(&self) -> bool {
        self.matcher.as_ref().is_some_and(GrammarMatcher::is_terminated)
    }
}
