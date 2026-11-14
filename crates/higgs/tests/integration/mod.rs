mod api_contract;
mod cli_contract;
mod cli_exec;
mod cli_serve;
mod error_contract;
mod proxy_e2e;
mod request_validation;
mod response_contract;
mod responses_e2e;
mod router;

/// The `higgs` binary Cargo built for these tests.
fn higgs_bin() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_higgs"))
}
