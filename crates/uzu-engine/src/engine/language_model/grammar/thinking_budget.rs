use tokenizers::Tokenizer;

#[derive(Clone, Copy)]
struct Snapshot {
    generated_len: usize,
    thinking_tokens: usize,
    end_matched: usize,
    force_index: Option<usize>,
    finished: bool,
}

/// Constrains a reasoning model to leave its thinking channel after a fixed
/// number of generated tokens. State is rollback-safe so speculative decoding
/// can explore and discard branches without losing the budget position.
pub(super) struct ThinkingBudget {
    budget: usize,
    end_sequence: Box<[u64]>,
    forced_sequence: Box<[u64]>,
    tokenizer: Tokenizer,
    generated_tokens: Vec<u32>,
    thinking_tokens: usize,
    end_matched: usize,
    force_index: Option<usize>,
    finished: bool,
    history: Vec<Snapshot>,
}

impl ThinkingBudget {
    pub fn new(
        budget: usize,
        end_sequence: Vec<u64>,
        forced_sequence: Vec<u64>,
        tokenizer: &Tokenizer,
    ) -> Self {
        debug_assert!(!end_sequence.is_empty());
        debug_assert!(!forced_sequence.is_empty());
        Self {
            budget,
            end_sequence: end_sequence.into_boxed_slice(),
            forced_sequence: forced_sequence.into_boxed_slice(),
            tokenizer: tokenizer.clone(),
            generated_tokens: Vec::new(),
            thinking_tokens: 0,
            end_matched: 0,
            force_index: None,
            finished: false,
            history: Vec::new(),
        }
    }

    pub fn next_forced_token(&mut self) -> Option<u64> {
        if self.finished {
            return None;
        }
        if let Some(index) = self.force_index {
            return self.forced_sequence.get(index).copied();
        }
        if self.thinking_tokens < self.budget || !self.is_utf8_boundary() {
            return None;
        }

        self.force_index = Some(0);
        self.forced_sequence.first().copied()
    }

    pub fn accept_token(
        &mut self,
        token_id: u64,
    ) -> bool {
        self.history.push(self.snapshot());
        if let Ok(token_id) = u32::try_from(token_id) {
            self.generated_tokens.push(token_id);
        }

        if let Some(index) = self.force_index {
            debug_assert_eq!(self.forced_sequence.get(index), Some(&token_id));
            let next_index = index + 1;
            if next_index == self.forced_sequence.len() {
                self.force_index = None;
                self.finished = true;
                return true;
            } else {
                self.force_index = Some(next_index);
            }
            return false;
        }

        self.end_matched = advance(&self.end_sequence, self.end_matched, token_id);
        if self.end_matched == self.end_sequence.len() {
            self.finished = true;
        } else {
            self.thinking_tokens += 1;
        }
        false
    }

    pub fn rollback(
        &mut self,
        num_tokens: usize,
    ) {
        for _ in 0..num_tokens {
            let Some(snapshot) = self.history.pop() else {
                break;
            };
            self.generated_tokens.truncate(snapshot.generated_len);
            self.thinking_tokens = snapshot.thinking_tokens;
            self.end_matched = snapshot.end_matched;
            self.force_index = snapshot.force_index;
            self.finished = snapshot.finished;
        }
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            generated_len: self.generated_tokens.len(),
            thinking_tokens: self.thinking_tokens,
            end_matched: self.end_matched,
            force_index: self.force_index,
            finished: self.finished,
        }
    }

    /// Byte-fallback tokenizers can end a token on an incomplete UTF-8 code
    /// point. Delay the forced marker until the decoder reaches a clean edge.
    fn is_utf8_boundary(&self) -> bool {
        self.tokenizer.decode(&self.generated_tokens, false).map_or(true, |text| !text.ends_with('\u{fffd}'))
    }
}

fn advance(
    sequence: &[u64],
    matched: usize,
    token_id: u64,
) -> usize {
    if matched == sequence.len() {
        return matched;
    }
    let mut length = matched + 1;
    while length > 0 {
        if sequence[length - 1] == token_id && sequence[matched + 1 - length..matched] == sequence[..length - 1] {
            return length;
        }
        length -= 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use tokenizers::{Tokenizer, models::wordlevel::WordLevel};
    use uzu_engine_macros::uzu_test;

    use super::ThinkingBudget;

    fn tokenizer() -> Tokenizer {
        Tokenizer::new(
            WordLevel::builder()
                .vocab(
                    [
                        ("<unk>".to_string(), 0),
                        ("thought".to_string(), 1),
                        ("</think>".to_string(), 2),
                        ("answer".to_string(), 3),
                    ]
                    .into_iter()
                    .collect(),
                )
                .unk_token("<unk>".to_string())
                .build()
                .unwrap(),
        )
    }

    #[uzu_test]
    fn forces_close_sequence_at_budget() {
        let mut budget = ThinkingBudget::new(2, vec![2], vec![2, 3], &tokenizer());
        assert_eq!(budget.next_forced_token(), None);
        let _ = budget.accept_token(1);
        assert_eq!(budget.next_forced_token(), None);
        let _ = budget.accept_token(1);
        assert_eq!(budget.next_forced_token(), Some(2));
        let _ = budget.accept_token(2);
        assert_eq!(budget.next_forced_token(), Some(3));
        assert!(budget.accept_token(3));
        assert_eq!(budget.next_forced_token(), None);
    }

    #[uzu_test]
    fn natural_close_disables_forcing() {
        let mut budget = ThinkingBudget::new(2, vec![2], vec![2, 3], &tokenizer());
        let _ = budget.accept_token(1);
        let _ = budget.accept_token(2);
        assert_eq!(budget.next_forced_token(), None);
    }

    #[uzu_test]
    fn rollback_restores_budget_and_forced_sequence() {
        let mut budget = ThinkingBudget::new(1, vec![2], vec![2, 3], &tokenizer());
        let _ = budget.accept_token(1);
        assert_eq!(budget.next_forced_token(), Some(2));
        let _ = budget.accept_token(2);
        assert_eq!(budget.next_forced_token(), Some(3));
        budget.rollback(1);
        assert_eq!(budget.next_forced_token(), Some(2));
        budget.rollback(1);
        assert_eq!(budget.next_forced_token(), None);
    }

    #[uzu_test]
    fn reports_the_pending_forced_token() {
        let mut budget = ThinkingBudget::new(1, vec![2], vec![2, 3], &tokenizer());
        let _ = budget.accept_token(1);
        assert_ne!(budget.next_forced_token(), Some(3));
        assert_eq!(budget.next_forced_token(), Some(2));
    }
}
