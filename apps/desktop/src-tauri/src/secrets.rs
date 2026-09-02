//! Secrets storage backed by the macOS Keychain via the `security` CLI.
//! Only a fixed, small set of secret names is accepted, so this cannot be
//! turned into a general-purpose keychain write/read primitive.

use std::process::Command;

const SERVICE: &str = "com.panbanda.higgs";
const ALLOWED_NAMES: &[&str] = &["api_key", "hf_token"];

/// `security`'s exit code for "the item could not be found".
const ERR_ITEM_NOT_FOUND: i32 = 44;

fn validate_name(name: &str) -> Result<(), String> {
    if ALLOWED_NAMES.contains(&name) {
        Ok(())
    } else {
        Err(format!("unknown secret name: {name}"))
    }
}

#[tauri::command]
pub fn secret_set(name: String, value: String) -> Result<(), String> {
    validate_name(&name)?;
    // The value is passed as a `Command` argument, never through a shell, so
    // it is not subject to shell interpolation. It is briefly visible to
    // other processes on the machine via `ps` while `security` runs; that is
    // the accepted trade-off over storing it in plaintext on disk.
    let output = Command::new("security")
        .args([
            "add-generic-password",
            "-U",
            "-a",
            &name,
            "-s",
            SERVICE,
            "-w",
            &value,
        ])
        .output()
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

#[tauri::command]
pub fn secret_get(name: String) -> Result<Option<String>, String> {
    validate_name(&name)?;
    let output = Command::new("security")
        .args(["find-generic-password", "-a", &name, "-s", SERVICE, "-w"])
        .output()
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        let value = String::from_utf8_lossy(&output.stdout)
            .trim_end_matches('\n')
            .to_owned();
        Ok(Some(value))
    } else if output.status.code() == Some(ERR_ITEM_NOT_FOUND) {
        Ok(None)
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

#[tauri::command]
pub fn secret_delete(name: String) -> Result<(), String> {
    validate_name(&name)?;
    let output = Command::new("security")
        .args(["delete-generic-password", "-a", &name, "-s", SERVICE])
        .output()
        .map_err(|error| error.to_string())?;
    if output.status.success() || output.status.code() == Some(ERR_ITEM_NOT_FOUND) {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_allowed_names() {
        assert!(validate_name("api_key").is_ok());
        assert!(validate_name("hf_token").is_ok());
    }

    #[test]
    fn rejects_unknown_names() {
        assert!(validate_name("").is_err());
        assert!(validate_name("password").is_err());
        assert!(validate_name("api_key; rm -rf /").is_err());
        assert!(validate_name("API_KEY").is_err());
    }
}
