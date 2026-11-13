//! Human-readable durations (`"90s"`, `"5m"`) and byte sizes (`"512MB"`, `"4GB"`).
//!
//! Sizes use binary multiples: `KB`, `MB`, `GB` and `TB` mean 1024-based units, exactly as the
//! `_mb` and `_gb` keys they replace did. The `KiB` spellings are accepted as synonyms.

use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use serde::de::{self, Unexpected, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// A parse failure with the text that failed and what was expected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct UnitError(String);

impl UnitError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Split `"1.5GB"` into the numeric part and the unit suffix.
fn split_number(text: &str) -> (&str, &str) {
    let end = text
        .find(|character: char| !(character.is_ascii_digit() || character == '.'))
        .unwrap_or(text.len());
    text.split_at(end)
}

/// A decimal number as an exact fraction, so `"1.5"` is 15 over 10 and never a rounded float.
struct Decimal {
    numerator: u128,
    denominator: u128,
}

impl Decimal {
    fn parse(text: &str) -> Option<Self> {
        let (whole, fraction) = text.split_once('.').unwrap_or((text, ""));
        if whole.is_empty() && fraction.is_empty() {
            return None;
        }
        if text.matches('.').count() > 1 {
            return None;
        }
        let digits = format!("{whole}{fraction}");
        let numerator = digits.parse::<u128>().ok()?;
        let denominator = 10_u128.checked_pow(u32::try_from(fraction.len()).ok()?)?;
        Some(Self {
            numerator,
            denominator,
        })
    }

    /// `self * unit` as a whole number, or `None` when it is fractional or overflows.
    fn times_exact(&self, unit: u128) -> Option<u128> {
        let product = self.numerator.checked_mul(unit)?;
        (product % self.denominator == 0).then_some(product / self.denominator)
    }
}

// -- Duration ---------------------------------------------------------------

/// A non-negative span of time written as a string such as `"90s"`, `"5m"` or `"1h30m"`.
///
/// Units: `ns`, `us`, `ms`, `s`, `m`, `h`, `d` (with `sec`, `min`, `hr`, `day` spellings). A bare
/// `0` is the only number accepted without a unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HumanDuration(Duration);

impl HumanDuration {
    pub const ZERO: Self = Self(Duration::ZERO);

    pub const fn new(duration: Duration) -> Self {
        Self(duration)
    }

    pub const fn get(self) -> Duration {
        self.0
    }

    pub const fn is_zero(self) -> bool {
        self.0.is_zero()
    }
}

impl From<Duration> for HumanDuration {
    fn from(duration: Duration) -> Self {
        Self(duration)
    }
}

fn duration_unit_nanos(unit: &str) -> Option<u128> {
    Some(match unit {
        "ns" => 1,
        "us" | "\u{b5}s" => 1_000,
        "ms" => 1_000_000,
        "s" | "sec" | "secs" | "second" | "seconds" => NANOS_PER_SECOND,
        "m" | "min" | "mins" | "minute" | "minutes" => 60 * NANOS_PER_SECOND,
        "h" | "hr" | "hrs" | "hour" | "hours" => 3600 * NANOS_PER_SECOND,
        "d" | "day" | "days" => 86_400 * NANOS_PER_SECOND,
        _ => return None,
    })
}

impl FromStr for HumanDuration {
    type Err = UnitError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let trimmed = text.trim();
        let hint = || {
            format!(
                "invalid duration {text:?}; write a number with a unit such as \"30s\", \"5m\" or \"1h\""
            )
        };
        if trimmed == "0" {
            return Ok(Self::ZERO);
        }
        if trimmed.is_empty() {
            return Err(UnitError::new(hint()));
        }
        let mut rest = trimmed;
        let mut total_nanos: u128 = 0;
        while !rest.is_empty() {
            let (number, after_number) = split_number(rest);
            let unit_end = after_number
                .find(|character: char| character.is_ascii_digit() || character == '.')
                .unwrap_or(after_number.len());
            let (unit, remainder) = after_number.split_at(unit_end);
            let decimal = Decimal::parse(number).ok_or_else(|| UnitError::new(hint()))?;
            let unit_nanos = duration_unit_nanos(unit.trim()).ok_or_else(|| {
                UnitError::new(format!(
                    "{}: unknown or missing unit {:?} (use ns, us, ms, s, m, h or d)",
                    hint(),
                    unit.trim()
                ))
            })?;
            let nanos = decimal.times_exact(unit_nanos).ok_or_else(|| {
                UnitError::new(format!(
                    "duration {text:?} is not a whole number of nanoseconds or is too large"
                ))
            })?;
            total_nanos = total_nanos
                .checked_add(nanos)
                .ok_or_else(|| UnitError::new(format!("duration {text:?} is too large")))?;
            rest = remainder.trim_start();
        }
        let seconds = u64::try_from(total_nanos / NANOS_PER_SECOND)
            .map_err(|_| UnitError::new(format!("duration {text:?} is too large")))?;
        let subsecond = u32::try_from(total_nanos % NANOS_PER_SECOND)
            .map_err(|_| UnitError::new(format!("duration {text:?} is too large")))?;
        Ok(Self(Duration::new(seconds, subsecond)))
    }
}

impl fmt::Display for HumanDuration {
    /// The largest unit that represents the value exactly: `300s` prints as `5m`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        const UNITS: [(&str, u128); 6] = [
            ("h", 3600 * NANOS_PER_SECOND),
            ("m", 60 * NANOS_PER_SECOND),
            ("s", NANOS_PER_SECOND),
            ("ms", 1_000_000),
            ("us", 1_000),
            ("ns", 1),
        ];
        let nanos = self.0.as_nanos();
        if nanos == 0 {
            return formatter.write_str("0s");
        }
        for (suffix, unit) in UNITS {
            if nanos % unit == 0 {
                return write!(formatter, "{}{suffix}", nanos / unit);
            }
        }
        write!(formatter, "{nanos}ns")
    }
}

impl Serialize for HumanDuration {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for HumanDuration {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct DurationVisitor;

        impl Visitor<'_> for DurationVisitor {
            type Value = HumanDuration;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a duration string such as \"30s\" or \"5m\"")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<HumanDuration, E> {
                value.parse().map_err(E::custom)
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<HumanDuration, E> {
                if value == 0 {
                    Ok(HumanDuration::ZERO)
                } else {
                    Err(E::custom(format!(
                        "a bare number {value} has no unit; write a string such as \"{value}s\""
                    )))
                }
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<HumanDuration, E> {
                u64::try_from(value)
                    .map_err(|_| E::invalid_value(Unexpected::Signed(value), &self))
                    .and_then(|unsigned| self.visit_u64(unsigned))
            }
        }

        deserializer.deserialize_any(DurationVisitor)
    }
}

// -- Byte size --------------------------------------------------------------

/// A size in bytes written as a string such as `"512MB"` or `"4GB"`, or as a bare integer of
/// bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ByteSize(u64);

const KIBIBYTE: u128 = 1024;

impl ByteSize {
    pub const fn from_bytes(bytes: u64) -> Self {
        Self(bytes)
    }

    pub const fn bytes(self) -> u64 {
        self.0
    }

    /// A whole number of mebibytes, for the legacy `_mb` keys.
    pub const fn from_mebibytes(mebibytes: u64) -> Self {
        Self(mebibytes.saturating_mul(1024 * 1024))
    }
}

fn size_unit_bytes(unit: &str) -> Option<u128> {
    Some(match unit.to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => KIBIBYTE,
        "m" | "mb" | "mib" => KIBIBYTE.pow(2),
        "g" | "gb" | "gib" => KIBIBYTE.pow(3),
        "t" | "tb" | "tib" => KIBIBYTE.pow(4),
        _ => return None,
    })
}

impl FromStr for ByteSize {
    type Err = UnitError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let trimmed = text.trim();
        let hint = || {
            format!(
                "invalid size {text:?}; write a number with a unit such as \"512MB\" or \"4GB\""
            )
        };
        let (number, unit) = split_number(trimmed);
        let decimal = Decimal::parse(number).ok_or_else(|| UnitError::new(hint()))?;
        let unit_bytes = size_unit_bytes(unit.trim()).ok_or_else(|| {
            UnitError::new(format!(
                "{}: unknown unit {:?} (use B, KB, MB, GB or TB; multiples are 1024-based)",
                hint(),
                unit.trim()
            ))
        })?;
        let bytes = decimal
            .times_exact(unit_bytes)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| {
                UnitError::new(format!(
                    "size {text:?} is not a whole number of bytes or is too large"
                ))
            })?;
        Ok(Self(bytes))
    }
}

impl fmt::Display for ByteSize {
    /// The largest unit that represents the value exactly: 4096 MiB prints as `4GB`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        const UNITS: [(&str, u64); 4] = [
            ("TB", 1 << 40),
            ("GB", 1 << 30),
            ("MB", 1 << 20),
            ("KB", 1 << 10),
        ];
        if self.0 == 0 {
            return formatter.write_str("0B");
        }
        for (suffix, unit) in UNITS {
            if self.0 % unit == 0 {
                return write!(formatter, "{}{suffix}", self.0 / unit);
            }
        }
        write!(formatter, "{}B", self.0)
    }
}

impl Serialize for ByteSize {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ByteSize {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct SizeVisitor;

        impl Visitor<'_> for SizeVisitor {
            type Value = ByteSize;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a size string such as \"512MB\" or a number of bytes")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<ByteSize, E> {
                value.parse().map_err(E::custom)
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<ByteSize, E> {
                Ok(ByteSize(value))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<ByteSize, E> {
                u64::try_from(value)
                    .map(ByteSize)
                    .map_err(|_| E::invalid_value(Unexpected::Signed(value), &self))
            }
        }

        deserializer.deserialize_any(SizeVisitor)
    }
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn duration(text: &str) -> Result<Duration, UnitError> {
        text.parse::<HumanDuration>().map(HumanDuration::get)
    }

    fn size(text: &str) -> Result<u64, UnitError> {
        text.parse::<ByteSize>().map(ByteSize::bytes)
    }

    #[test]
    fn durations_parse_every_unit() {
        assert_eq!(duration("250ms").unwrap(), Duration::from_millis(250));
        assert_eq!(duration("90s").unwrap(), Duration::from_secs(90));
        assert_eq!(duration("5m").unwrap(), Duration::from_secs(300));
        assert_eq!(duration("2h").unwrap(), Duration::from_secs(7200));
        assert_eq!(duration("1d").unwrap(), Duration::from_secs(86_400));
        assert_eq!(duration("15us").unwrap(), Duration::from_micros(15));
        assert_eq!(duration("7ns").unwrap(), Duration::from_nanos(7));
    }

    #[test]
    fn durations_accept_long_spellings_decimals_and_compounds() {
        assert_eq!(duration("30 seconds").unwrap(), Duration::from_secs(30));
        assert_eq!(duration("10min").unwrap(), Duration::from_secs(600));
        assert_eq!(duration("1.5h").unwrap(), Duration::from_secs(5400));
        assert_eq!(duration("1h30m").unwrap(), Duration::from_secs(5400));
        assert_eq!(duration(" 2s ").unwrap(), Duration::from_secs(2));
    }

    #[test]
    fn bare_zero_is_the_only_unitless_duration() {
        assert_eq!(duration("0").unwrap(), Duration::ZERO);
        assert_eq!(duration("0s").unwrap(), Duration::ZERO);
        assert!(duration("30").is_err());
    }

    #[test]
    fn invalid_durations_are_rejected_with_a_hint() {
        for bad in [
            "",
            "abc",
            "5x",
            "-5s",
            "1.5.2s",
            "s",
            "1ns2",
            "0.0000000001s",
        ] {
            let error = duration(bad).unwrap_err().to_string();
            assert!(error.contains("duration"), "{bad}: {error}");
        }
        assert!(
            duration("5 lightyears")
                .unwrap_err()
                .to_string()
                .contains("unit")
        );
        assert!(duration("99999999999999999999999d").is_err());
    }

    #[test]
    fn duration_formatting_uses_the_largest_exact_unit_and_round_trips() {
        for (text, rendered) in [
            ("300s", "5m"),
            ("90s", "90s"),
            ("3600s", "1h"),
            ("2500ms", "2500ms"),
            ("1500us", "1500us"),
            ("0", "0s"),
            ("86400s", "24h"),
        ] {
            let parsed: HumanDuration = text.parse().unwrap();
            assert_eq!(parsed.to_string(), rendered);
            assert_eq!(rendered.parse::<HumanDuration>().unwrap(), parsed);
        }
    }

    #[test]
    fn sizes_parse_binary_units_and_synonyms() {
        assert_eq!(size("512MB").unwrap(), 512 * 1024 * 1024);
        assert_eq!(size("4GB").unwrap(), 4 * 1024 * 1024 * 1024);
        assert_eq!(size("4GiB").unwrap(), 4 * 1024 * 1024 * 1024);
        assert_eq!(size("1tb").unwrap(), 1 << 40);
        assert_eq!(size("64k").unwrap(), 64 * 1024);
        assert_eq!(size("100").unwrap(), 100);
        assert_eq!(size("100B").unwrap(), 100);
        assert_eq!(size("1.5GB").unwrap(), 3 * (1 << 29));
        assert_eq!(size(" 2 MB ").unwrap(), 2 * 1024 * 1024);
    }

    #[test]
    fn invalid_sizes_are_rejected() {
        for bad in [
            "",
            "MB",
            "5XB",
            "-1MB",
            "1.5.5GB",
            "0.0001KB",
            "99999999999999999999TB",
        ] {
            assert!(size(bad).is_err(), "{bad} should be rejected");
        }
        assert!(size("5XB").unwrap_err().to_string().contains("unit"));
    }

    #[test]
    fn size_formatting_uses_the_largest_exact_unit_and_round_trips() {
        for (bytes, rendered) in [
            (4_u64 << 30, "4GB"),
            (10 << 20, "10MB"),
            (1536 << 20, "1536MB"),
            (1000, "1000B"),
            (0, "0B"),
            (3 << 10, "3KB"),
            (2 << 40, "2TB"),
        ] {
            let value = ByteSize::from_bytes(bytes);
            assert_eq!(value.to_string(), rendered);
            assert_eq!(rendered.parse::<ByteSize>().unwrap(), value);
        }
    }

    #[test]
    fn serde_accepts_strings_and_bare_integers() {
        #[derive(Debug, Deserialize)]
        struct Holder {
            wait: HumanDuration,
            limit: ByteSize,
        }
        let holder: Holder = toml_edit_free_parse(r#"{"wait": "2m", "limit": 1048576}"#);
        assert_eq!(holder.wait.get(), Duration::from_secs(120));
        assert_eq!(holder.limit.bytes(), 1_048_576);

        let zero: Holder = toml_edit_free_parse(r#"{"wait": 0, "limit": "1MB"}"#);
        assert!(zero.wait.is_zero());

        let bare = serde_json::from_str::<Holder>(r#"{"wait": 30, "limit": "1MB"}"#);
        assert!(bare.unwrap_err().to_string().contains("no unit"));
        let negative = serde_json::from_str::<Holder>(r#"{"wait": "1s", "limit": -1}"#);
        assert!(negative.is_err());
    }

    fn toml_edit_free_parse<T: serde::de::DeserializeOwned>(json: &str) -> T {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn serde_serializes_as_strings() {
        assert_eq!(
            serde_json::to_string(&HumanDuration::new(Duration::from_secs(300))).unwrap(),
            "\"5m\""
        );
        assert_eq!(
            serde_json::to_string(&ByteSize::from_bytes(4 << 30)).unwrap(),
            "\"4GB\""
        );
    }
}
