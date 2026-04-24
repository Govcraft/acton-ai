//! Verifies the `chat` subcommand surfaces `--skill-dir` through clap.

use acton_ai::cli::Cli;
use clap::Parser;

#[test]
fn chat_help_mentions_skill_dir() {
    // clap emits --help as a parse error; matching that error kind confirms
    // the subcommand is recognized and we reached its help renderer.
    let err = Cli::try_parse_from(["acton-ai", "chat", "--help"])
        .expect_err("--help always returns Err for printing");
    assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp);
    let rendered = err.to_string();
    assert!(
        rendered.contains("skill-dir"),
        "chat --help should mention --skill-dir, got:\n{rendered}"
    );
}

#[test]
fn chat_accepts_repeated_skill_dir() {
    let parsed =
        Cli::try_parse_from(["acton-ai", "chat", "--skill-dir", "./a", "--skill-dir", "./b"])
            .expect("parse should succeed with valid args");
    match parsed.command {
        acton_ai::cli::Commands::Chat(args) => {
            assert_eq!(args.skill_dirs.len(), 2);
            assert_eq!(args.skill_dirs[0].to_str(), Some("./a"));
            assert_eq!(args.skill_dirs[1].to_str(), Some("./b"));
        }
        other => panic!("expected Commands::Chat, got {other:?}"),
    }
}
