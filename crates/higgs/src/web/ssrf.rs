//! Address policy and name resolution for outbound fetches.

use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use futures::future::BoxFuture;

/// Whether `address` is outside the globally routable unicast range.
///
/// Covers loopback, private, link-local (including cloud metadata endpoints),
/// CGNAT, multicast, documentation and other reserved ranges. IPv6 forms that
/// embed an IPv4 address are judged by the embedded address, so
/// `::ffff:127.0.0.1` cannot slip past the IPv4 rules.
pub fn is_non_public(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(v4) => is_non_public_v4(v4),
        IpAddr::V6(v6) => is_non_public_v6(v6),
    }
}

fn is_non_public_v4(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    a == 0
        || a == 10
        || a == 127
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 192 && b == 168)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 224
}

fn is_non_public_v6(address: Ipv6Addr) -> bool {
    let [s0, s1, s2, s3, s4, s5, s6, s7] = address.segments();
    let embedded_v4 = |high: u16, low: u16| {
        let [h1, h2] = high.to_be_bytes();
        let [l1, l2] = low.to_be_bytes();
        is_non_public_v4(Ipv4Addr::new(h1, h2, l1, l2))
    };
    if address.is_unspecified() || address.is_loopback() {
        return true;
    }
    // IPv4-mapped (::ffff:a.b.c.d) and the deprecated IPv4-compatible (::a.b.c.d).
    if s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 && s4 == 0 && (s5 == 0xffff || s5 == 0) {
        return embedded_v4(s6, s7);
    }
    // NAT64 (64:ff9b::/96) carries an IPv4 address in the low 32 bits.
    if s0 == 0x0064 && s1 == 0xff9b && s2 == 0 && s3 == 0 && s4 == 0 && s5 == 0 {
        return embedded_v4(s6, s7);
    }
    // 6to4 (2002::/16) carries one in bits 16..48.
    if s0 == 0x2002 {
        return embedded_v4(s1, s2);
    }
    (s0 & 0xfe00) == 0xfc00 // unique local fc00::/7
        || (s0 & 0xffc0) == 0xfe80 // link-local fe80::/10
        || (s0 & 0xffc0) == 0xfec0 // deprecated site-local fec0::/10
        || (s0 & 0xff00) == 0xff00 // multicast
        || (s0 == 0x2001 && s1 == 0x0db8) // documentation
        || (s0 == 0x2001 && s1 == 0) // Teredo
        || (s0 == 0x0100 && s1 == 0 && s2 == 0 && s3 == 0) // discard-only
}

#[derive(Debug, Clone, Default)]
pub struct AddressPolicy {
    pub allow_private: bool,
    /// Lets tests reach a mock server on loopback while every other
    /// non-public address stays blocked.
    #[cfg(test)]
    pub exempt: Vec<IpAddr>,
}

impl AddressPolicy {
    pub fn check(&self, address: IpAddr) -> Result<(), String> {
        if self.allow_private || !is_non_public(address) {
            return Ok(());
        }
        #[cfg(test)]
        if self.exempt.contains(&address) {
            return Ok(());
        }
        Err(format!(
            "{address} is not a public address; fetching loopback, private and link-local addresses is disabled"
        ))
    }
}

pub trait Resolver: Send + Sync {
    fn resolve<'a>(
        &'a self,
        host: &'a str,
        port: u16,
    ) -> BoxFuture<'a, io::Result<Vec<SocketAddr>>>;
}

pub struct SystemResolver;

impl Resolver for SystemResolver {
    fn resolve<'a>(
        &'a self,
        host: &'a str,
        port: u16,
    ) -> BoxFuture<'a, io::Result<Vec<SocketAddr>>> {
        Box::pin(async move { Ok(tokio::net::lookup_host((host, port)).await?.collect()) })
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    fn blocked(text: &str) -> bool {
        is_non_public(text.parse().unwrap())
    }

    #[test]
    fn ipv4_private_loopback_and_reserved_ranges_are_blocked() {
        for address in [
            "0.0.0.0",
            "0.1.2.3",
            "10.0.0.1",
            "10.255.255.255",
            "127.0.0.1",
            "127.255.255.254",
            "100.64.0.1",
            "100.127.255.255",
            "169.254.169.254",
            "172.16.0.1",
            "172.31.255.255",
            "192.0.0.8",
            "192.0.2.1",
            "192.168.1.1",
            "198.18.0.1",
            "198.19.255.255",
            "198.51.100.7",
            "203.0.113.9",
            "224.0.0.1",
            "239.255.255.250",
            "240.0.0.1",
            "255.255.255.255",
        ] {
            assert!(blocked(address), "{address} should be blocked");
        }
    }

    #[test]
    fn ipv4_public_addresses_are_allowed_including_range_edges() {
        for address in [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
            "100.63.255.255",
            "100.128.0.1",
            "172.15.255.255",
            "172.32.0.1",
            "198.17.255.255",
            "198.20.0.1",
            "223.255.255.255",
        ] {
            assert!(!blocked(address), "{address} should be allowed");
        }
    }

    #[test]
    fn ipv6_special_ranges_are_blocked() {
        for address in [
            "::",
            "::1",
            "fc00::1",
            "fd12:3456:789a::1",
            "fe80::1",
            "febf::1",
            "fec0::1",
            "ff02::1",
            "2001:db8::1",
            "2001:0:4136:e378:8000:63bf:3fff:fdd2",
            "100::1",
        ] {
            assert!(blocked(address), "{address} should be blocked");
        }
    }

    #[test]
    fn ipv6_forms_embedding_ipv4_are_judged_by_the_embedded_address() {
        for address in [
            "::ffff:127.0.0.1",
            "::ffff:7f00:1",
            "::ffff:10.0.0.1",
            "::ffff:169.254.169.254",
            "::127.0.0.1",
            "64:ff9b::7f00:1",
            "64:ff9b::a9fe:a9fe",
            "2002:7f00:1::",
            "2002:c0a8:101::1",
        ] {
            assert!(blocked(address), "{address} should be blocked");
        }
        for address in ["::ffff:8.8.8.8", "64:ff9b::808:808", "2002:808:808::1"] {
            assert!(!blocked(address), "{address} should be allowed");
        }
    }

    #[test]
    fn ipv6_public_addresses_are_allowed() {
        for address in [
            "2606:4700:4700::1111",
            "2001:4860:4860::8888",
            "2a00:1450:4001::200e",
        ] {
            assert!(!blocked(address), "{address} should be allowed");
        }
    }

    #[test]
    fn policy_can_allow_private_addresses_wholesale() {
        let strict = AddressPolicy::default();
        let permissive = AddressPolicy {
            allow_private: true,
            ..AddressPolicy::default()
        };
        let loopback: IpAddr = "127.0.0.1".parse().unwrap();
        assert!(
            strict
                .check(loopback)
                .unwrap_err()
                .contains("not a public address")
        );
        assert!(permissive.check(loopback).is_ok());
        assert!(strict.check("8.8.8.8".parse().unwrap()).is_ok());
    }

    #[test]
    fn url_parsing_normalizes_numeric_host_tricks_to_addresses_that_are_blocked() {
        for text in [
            "http://2130706433/",
            "http://0x7f.1/",
            "http://0177.0.0.1/",
            "http://127.1/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[0:0:0:0:0:ffff:7f00:1]/",
        ] {
            let url = reqwest::Url::parse(text).unwrap();
            let host = url
                .host_str()
                .unwrap()
                .trim_matches(|c| c == '[' || c == ']');
            let address: IpAddr = host
                .parse()
                .unwrap_or_else(|_| panic!("{text} did not normalize to an address: {host}"));
            assert!(is_non_public(address), "{text} -> {address}");
        }
    }
}
