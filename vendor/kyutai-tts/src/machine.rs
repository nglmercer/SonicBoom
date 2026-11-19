//! Text scheduling for the delayed-streams TTS: a port of `StateMachine`,
//! `Entry` and `script_to_entries` from `moshi/models/tts.py`.
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct TokenIds {
    /// Text cardinality including the initial token (`text_card + 1`).
    pub card: i64,
    pub new_word: i64,
    pub pad: i64,
    pub main: i64,
    pub other: i64,
}

impl TokenIds {
    pub fn new(card: i64) -> Self {
        Self { card, new_word: 0, pad: 3, main: 1, other: 2 }
    }
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub tokens: Vec<i64>,
    pub text: String,
    pub padding: usize,
}

#[derive(Debug)]
pub struct State {
    pub entries: VecDeque<Entry>,
    pub remaining_padding: i64,
    pub forced_padding: i64,
    pub queued: VecDeque<i64>,
    pub lookahead_queued: VecDeque<i64>,
    pub end_step: Option<usize>,
    pub consumption_times: Vec<usize>,
    pub transcript: Vec<(String, usize)>,
}

impl State {
    fn tokens_ahead(&self, mut lookahead: usize) -> Vec<i64> {
        for e in &self.entries {
            if !e.tokens.is_empty() {
                lookahead -= 1;
                if lookahead == 0 {
                    return e.tokens.clone();
                }
            }
        }
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct StateMachine {
    pub ids: TokenIds,
    pub second_stream_ahead: usize,
    pub max_padding: i64,
    pub initial_padding: i64,
}

impl StateMachine {
    pub fn new_state(&self, entries: Vec<Entry>) -> State {
        State {
            entries: entries.into(),
            remaining_padding: self.initial_padding,
            forced_padding: self.initial_padding,
            queued: VecDeque::new(),
            lookahead_queued: VecDeque::new(),
            end_step: None,
            consumption_times: vec![],
            transcript: vec![],
        }
    }

    /// Returns the value to feed as the text input at the next step (already
    /// multiplexed with the lookahead stream) and whether a word was consumed.
    pub fn process(&self, step: usize, state: &mut State, token: i64) -> (i64, bool) {
        let ids = self.ids;
        let mut consumed = false;
        let mut token = token;
        if token != ids.new_word && token != ids.pad {
            token = ids.pad;
        }
        if !state.queued.is_empty() {
            token = ids.pad;
        } else if state.forced_padding > 0 {
            token = ids.pad;
        } else if state.remaining_padding <= 0 {
            token = ids.new_word;
        }

        if token == ids.new_word {
            if let Some(entry) = state.entries.pop_front() {
                state.consumption_times.push(step);
                consumed = true;
                if !entry.tokens.is_empty() {
                    state.transcript.push((entry.text.clone(), step));
                    state.queued.extend(entry.tokens.iter().copied());
                    if self.second_stream_ahead > 0 {
                        let ahead = state.tokens_ahead(self.second_stream_ahead);
                        state.lookahead_queued.extend(ahead);
                    }
                    state.remaining_padding = self.max_padding;
                } else {
                    token = ids.pad;
                }
                state.forced_padding = entry.padding as i64;
            } else {
                token = ids.pad;
                if self.second_stream_ahead > 0 && state.end_step.is_none() {
                    token = ids.new_word;
                }
                if state.end_step.is_none() {
                    state.end_step = Some(step);
                }
            }
        }

        let mut output = if token == ids.pad {
            if state.remaining_padding > 0 {
                state.remaining_padding -= 1;
            }
            if state.forced_padding > 0 {
                state.forced_padding -= 1;
            }
            state.queued.pop_front().unwrap_or(ids.pad)
        } else {
            ids.new_word
        };

        if self.second_stream_ahead > 0 {
            let mut second = -1;
            if output == ids.new_word {
                second = ids.new_word;
                output = state.queued.pop_front().unwrap_or(ids.pad);
            } else if let Some(t) = state.lookahead_queued.pop_front() {
                second = t;
            }
            output += (second + 1) * ids.card;
        }
        (output, consumed)
    }
}

/// Split `text` into word entries (single speaker, `multi_speaker = true`), as
/// `script_to_entries` does for a one-turn script without `<break>` tags.
pub fn script_to_entries(
    encode: &dyn Fn(&str) -> Vec<i64>,
    ids: &TokenIds,
    text: &str,
    padding_between: usize,
) -> Vec<Entry> {
    let line = text
        .replace('’', "'")
        .replace(':', " ")
        .replace(['(', ')'], "");
    let mut entries = Vec::new();
    let mut first_content = true;
    for word in line.split_whitespace() {
        let mut tokens = encode(word);
        if first_content {
            // The first word of a turn carries the speaker token.
            tokens.insert(0, ids.main);
            first_content = false;
        }
        let padding = if padding_between > 0 {
            (padding_between + tokens.len()).saturating_sub(1)
        } else {
            0
        };
        entries.push(Entry { tokens, text: word.to_string(), padding });
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> StateMachine {
        StateMachine { ids: TokenIds::new(8001), second_stream_ahead: 2, max_padding: 8, initial_padding: 2 }
    }

    #[test]
    fn script_adds_speaker_token_and_padding() {
        let ids = TokenIds::new(8001);
        let enc = |w: &str| vec![100 + w.len() as i64, 200];
        let e = script_to_entries(&enc, &ids, "Hello, world: ok", 1);
        assert_eq!(e.len(), 3);
        assert_eq!(e[0].tokens[0], 1, "first word carries the main speaker token");
        assert_eq!(e[1].tokens, vec![105, 200]);
        assert_eq!(e[0].padding, 3); // 1 + len(3 tokens) - 1
        assert_eq!(e[1].padding, 2);
    }

    #[test]
    fn forced_padding_first_then_new_word_consumes_entry() {
        let m = machine();
        let mut st = m.new_state(vec![Entry { tokens: vec![7, 8], text: "a".into(), padding: 0 }]);
        // initial_padding = 2: the first two steps must pad whatever the model says.
        let (o0, c0) = m.process(0, &mut st, m.ids.new_word);
        let (o1, c1) = m.process(1, &mut st, m.ids.new_word);
        assert!(!c0 && !c1);
        assert_eq!(o0 % m.ids.card, m.ids.pad);
        assert_eq!(o1 % m.ids.card, m.ids.pad);
        // now a new word is allowed; it is consumed and its tokens queued.
        let (_o2, c2) = m.process(2, &mut st, m.ids.new_word);
        assert!(c2);
        assert_eq!(st.transcript, vec![("a".to_string(), 2)]);
        // the first queued token is delivered through the main stream right away.
        assert_eq!(_o2 % m.ids.card, 7);
    }

    #[test]
    fn end_step_is_set_when_entries_run_out() {
        let m = machine();
        let mut st = m.new_state(vec![]);
        for s in 0..2 {
            m.process(s, &mut st, m.ids.pad);
        }
        assert!(st.end_step.is_none());
        m.process(2, &mut st, m.ids.new_word);
        assert_eq!(st.end_step, Some(2));
    }
}
