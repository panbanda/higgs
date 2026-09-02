//! Secrets storage backed by the macOS Keychain through the Security
//! framework, so a credential never appears in a process argument list or on
//! disk in plain text. Only a fixed, small set of secret names is accepted.

use security_framework::base::Error as KeychainError;
use security_framework::passwords::{
    delete_generic_password, get_generic_password, set_generic_password,
};

const SERVICE: &str = "com.panbanda.higgs";
const ALLOWED_NAMES: &[&str] = &["api_key", "hf_token"];

/// `errSecItemNotFound` from Security.framework.
const ERR_ITEM_NOT_FOUND: i32 = -25300;

fn validate_name(name: &str) -> Result<(), String> {
    if ALLOWED_NAMES.contains(&name) {
        Ok(())
    } else {
        Err(format!("unknown secret name: {name}"))
    }
}

fn is_not_found(error: &KeychainError) -> bool {
    error.code() == ERR_ITEM_NOT_FOUND
}

#[tauri::command]
pub fn secret_set(name: String, value: String) -> Result<(), String> {
    validate_name(&name)?;
    set_generic_password(SERVICE, &name, value.as_bytes()).map_err(|error| error.to_string())
}

#[tauri::command]
pub fn secret_get(name: String) -> Result<Option<String>, String> {
    validate_name(&name)?;
    match get_generic_password(SERVICE, &name) {
        Ok(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
        Err(error) if is_not_found(&error) => Ok(None),
        Err(error) => Err(error.to_string()),
    }
}

#[tauri::command]
pub fn secret_delete(name: String) -> Result<(), String> {
    validate_name(&name)?;
    match delete_generic_password(SERVICE, &name) {
        Ok(()) => Ok(()),
        Err(error) if is_not_found(&error) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_known_names() {
        assert!(validate_name("api_key").is_ok());
        assert!(validate_name("hf_token").is_ok());
        assert!(validate_name("password").is_err());
        assert!(validate_name("").is_err());
    }
}
