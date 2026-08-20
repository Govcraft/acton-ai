//! Incremental line assembly for server-sent-event streams.
//!
//! An SSE response arrives as arbitrary network chunks, and a chunk boundary
//! can land in the middle of a `data:` line — or in the middle of a
//! multi-byte UTF-8 character. Splitting each chunk with `str::lines` in
//! isolation therefore corrupts any event that straddles a boundary: the
//! first half fails to parse and the second half, missing its `data: `
//! prefix, is dropped. This module owns the carry-over: bytes go in, only
//! complete lines come out, and the unterminated tail waits for the next
//! chunk (or the end of the stream).

use crate::llm::error::LLMError;
use futures::stream::BoxStream;
use futures::{Stream, StreamExt};

/// Reassembles complete lines from arbitrarily split byte chunks.
#[derive(Debug, Default)]
pub(crate) struct LineAssembler {
    /// Bytes received but not yet terminated by a newline.
    tail: Vec<u8>,
}

impl LineAssembler {
    /// Feeds one network chunk and returns every line it completed.
    ///
    /// Lines are split on `\n`; a trailing `\r` is stripped. Bytes after the
    /// last newline stay buffered until a later chunk terminates them.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.tail.extend_from_slice(bytes);
        let Some(last_newline) = self.tail.iter().rposition(|&b| b == b'\n') else {
            return Vec::new();
        };
        let rest = self.tail.split_off(last_newline + 1);
        let complete = std::mem::replace(&mut self.tail, rest);
        complete[..complete.len() - 1]
            .split(|&b| b == b'\n')
            .map(|line| {
                let line = line.strip_suffix(b"\r").unwrap_or(line);
                String::from_utf8_lossy(line).into_owned()
            })
            .collect()
    }

    /// Takes whatever the stream left unterminated when it ended.
    ///
    /// Some servers close the connection right after the final event without
    /// a trailing newline; that event still has to be parsed.
    pub(crate) fn take_remainder(&mut self) -> Option<String> {
        if self.tail.is_empty() {
            return None;
        }
        let line = String::from_utf8_lossy(&self.tail).into_owned();
        self.tail.clear();
        Some(line)
    }
}

/// Adapts a byte-chunk stream into a stream of complete-line batches.
///
/// Each item is the batch of lines one network chunk completed (possibly
/// empty, when the chunk ended mid-line); the final batch carries any
/// unterminated remainder. Transport errors surface as
/// [`LLMError::stream_error`] items and leave the stream running, matching
/// how both clients treated them before.
pub(crate) fn lines<S, B, E>(bytes: S) -> BoxStream<'static, Result<Vec<String>, LLMError>>
where
    S: Stream<Item = Result<B, E>> + Send + Unpin + 'static,
    B: AsRef<[u8]>,
    E: std::fmt::Display,
{
    futures::stream::unfold(
        (bytes, LineAssembler::default(), false),
        |(mut bytes, mut assembler, done)| async move {
            if done {
                return None;
            }
            match bytes.next().await {
                Some(Ok(chunk)) => {
                    let batch = assembler.push(chunk.as_ref());
                    Some((Ok(batch), (bytes, assembler, false)))
                }
                Some(Err(e)) => Some((
                    Err(LLMError::stream_error(format!("stream read error: {}", e))),
                    (bytes, assembler, false),
                )),
                None => {
                    let batch = assembler.take_remainder().into_iter().collect();
                    Some((Ok(batch), (bytes, assembler, true)))
                }
            }
        },
    )
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_split_across_chunks_reassembles() {
        let mut assembler = LineAssembler::default();
        assert!(assembler.push(b"data: {\"text\": \"hel").is_empty());
        let lines = assembler.push(b"lo\"}\n");
        assert_eq!(lines, vec!["data: {\"text\": \"hello\"}"]);
    }

    #[test]
    fn a_utf8_character_split_across_chunks_reassembles() {
        let text = "data: café\n".as_bytes();
        // Split inside the two-byte 'é'.
        let split = text.len() - 2;
        let mut assembler = LineAssembler::default();
        assert!(assembler.push(&text[..split]).is_empty());
        assert_eq!(assembler.push(&text[split..]), vec!["data: café"]);
    }

    #[test]
    fn multiple_lines_in_one_chunk_all_come_out() {
        let mut assembler = LineAssembler::default();
        let lines = assembler.push(b"data: a\n\ndata: b\ndata: c");
        assert_eq!(lines, vec!["data: a", "", "data: b"]);
        assert_eq!(assembler.push(b"\n"), vec!["data: c"]);
    }

    #[test]
    fn crlf_terminators_are_stripped() {
        let mut assembler = LineAssembler::default();
        assert_eq!(assembler.push(b"data: a\r\ndata: b\r\n"), vec![
            "data: a", "data: b"
        ]);
    }

    #[test]
    fn the_remainder_surfaces_after_the_stream_ends() {
        let mut assembler = LineAssembler::default();
        assert!(assembler.push(b"data: [DONE]").is_empty());
        assert_eq!(assembler.take_remainder().as_deref(), Some("data: [DONE]"));
        assert_eq!(assembler.take_remainder(), None);
    }

    #[tokio::test]
    async fn the_line_stream_reassembles_and_flushes_the_tail() {
        let chunks: Vec<Result<&[u8], std::convert::Infallible>> = vec![
            Ok(b"data: {\"a\":".as_slice()),
            Ok(b" 1}\ndata: [DO".as_slice()),
            Ok(b"NE]".as_slice()),
        ];
        let collected: Vec<_> = lines(futures::stream::iter(chunks)).collect().await;
        let flat: Vec<String> = collected
            .into_iter()
            .flat_map(|batch| batch.expect("no transport errors"))
            .collect();
        assert_eq!(flat, vec!["data: {\"a\": 1}", "data: [DONE]"]);
    }
}
