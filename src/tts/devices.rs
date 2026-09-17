//! Audio output device discovery and selection types.
//!
//! All device manipulation runs inside the dedicated playback thread (see
//! [`crate::tts::queue`]); HTTP handlers only exchange these plain data
//! types with it. The pure helpers ([`build_device_list`],
//! [`match_hardware`]) carry the selection semantics so they are unit
//! testable without audio hardware.
//!
//! Device names are backend/OS dependent: CPAL exposes display names only
//! (e.g. `Speakers (Realtek(R) Audio)`), not stable identifiers, so `id` is
//! the OS-reported name. When the backend reports duplicate names the
//! duplicates gain ` #2`, ` #3`, ... suffixes on `id` (best-effort: suffix
//! assignment follows enumeration order).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Logical selection that follows the OS default output device.
pub const DEFAULT_DEVICE_ID: &str = "default";

/// True when `selection` requests OS-default behavior (case-insensitive,
/// surrounding whitespace ignored).
pub fn is_default_selection(selection: &str) -> bool {
    selection.trim().eq_ignore_ascii_case(DEFAULT_DEVICE_ID)
}

/// Normalize a raw selection: surrounding whitespace is trimmed, blank
/// input and any casing of `default` become canonical `"default"`.
pub fn normalize_selection(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case(DEFAULT_DEVICE_ID) {
        DEFAULT_DEVICE_ID.to_string()
    } else {
        trimmed.to_string()
    }
}

/// One selectable output device (option-source friendly: `id` + `name`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputDeviceInfo {
    /// Selectable identifier: the OS-reported name, or `default` for the
    /// logical entry. Duplicate names gain ` #N` suffixes.
    pub id: String,
    /// Display name as reported by the OS backend.
    pub name: String,
    /// True for the logical `default` entry and for the hardware entry (if
    /// any) that is currently the OS default output.
    pub is_default: bool,
    /// True when this entry matches the current selection.
    pub is_selected: bool,
}

/// Full device list plus the configured selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputDeviceList {
    pub devices: Vec<OutputDeviceInfo>,
    /// Configured selection (`default` or an explicit id). It may name a
    /// device that is currently missing; that entry then simply matches
    /// nothing in `devices`.
    pub selected: String,
}

/// Configured selection versus live hardware state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveOutputDevice {
    /// Configured selection (`default` or an explicit device id).
    pub device: String,
    /// Currently resolved hardware device name, when an output stream is
    /// active. `None` while the selection is unavailable (missing device,
    /// no stream yet) — never silently another device.
    pub resolved_name: Option<String>,
    /// Whether an output stream is currently active for the selection.
    pub available: bool,
}

/// Assign unique selectable ids to enumerated hardware names (pure).
/// First occurrences keep their bare name; later duplicates gain ` #N`
/// suffixes, skipping candidates that collide with real names.
fn disambiguated_ids(hardware_names: &[String]) -> Vec<(String, String)> {
    let mut ids = Vec::with_capacity(hardware_names.len());
    let mut used: HashSet<String> = HashSet::new();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for name in hardware_names {
        let count = counts.entry(name.as_str()).or_insert(0);
        *count += 1;
        let mut n = *count;
        let mut candidate = if n == 1 {
            name.clone()
        } else {
            format!("{name} #{n}")
        };
        while used.contains(&candidate) {
            n += 1;
            candidate = format!("{name} #{n}");
        }
        *count = n;
        used.insert(candidate.clone());
        ids.push((candidate, name.clone()));
    }
    ids
}

/// Build the API device list from enumerated hardware names (pure).
///
/// - Always exposes the logical `"default"` entry first (`System Default`).
/// - `default_name` marks which hardware entry (if any) is the OS default.
/// - `selected` marks the matching entry; a missing explicit selection
///   matches nothing (the `selected` field still reports it).
pub fn build_device_list(
    hardware_names: Vec<String>,
    default_name: Option<&str>,
    selected: &str,
) -> OutputDeviceList {
    let selected_is_default = is_default_selection(selected);
    let mut devices = Vec::with_capacity(hardware_names.len() + 1);
    devices.push(OutputDeviceInfo {
        id: DEFAULT_DEVICE_ID.to_string(),
        name: "System Default".to_string(),
        is_default: true,
        is_selected: selected_is_default,
    });
    for (id, name) in disambiguated_ids(&hardware_names) {
        devices.push(OutputDeviceInfo {
            is_default: default_name.is_some_and(|d| d == name),
            is_selected: !selected_is_default && selected == id,
            id,
            name,
        });
    }
    OutputDeviceList {
        devices,
        selected: selected.to_string(),
    }
}

/// A `selected` value resolved against enumerated hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceMatch {
    /// Open the OS default output.
    Default,
    /// Open the 1-based `occurrence`-th device named `name`.
    Named { name: String, occurrence: usize },
}

/// Match a normalized `selected` value against enumerated hardware (pure).
///
/// - `"default"` (any casing) → [`DeviceMatch::Default`].
/// - exact id match (bare names and ` #N` duplicates) →
///   [`DeviceMatch::Named`].
/// - otherwise → `None`: unknown device (the caller reports `400` and must
///   not fall back to another device).
pub fn match_hardware(selected: &str, hardware_names: &[String]) -> Option<DeviceMatch> {
    if is_default_selection(selected) {
        return Some(DeviceMatch::Default);
    }
    let ids = disambiguated_ids(hardware_names);
    let pos = ids.iter().position(|(id, _)| id == selected)?;
    let name = ids[pos].1.clone();
    let occurrence = ids[..pos].iter().filter(|(_, n)| *n == name).count() + 1;
    Some(DeviceMatch::Named { name, occurrence })
}

/// Enumerate output device names from the OS backend (audio thread only).
///
/// Returns `(hardware_names, default_name)`. Output devices only — never
/// input-only devices. Devices whose names cannot be read are skipped.
/// Enumeration failure is an error (the API reports `500`, never an empty
/// success); zero devices is a valid empty list.
pub fn enumerate_host_devices() -> Result<(Vec<String>, Option<String>), String> {
    use rodio::cpal::traits::{DeviceTrait, HostTrait};

    let host = rodio::cpal::default_host();
    let default_name = host.default_output_device().and_then(|d| d.name().ok());
    let devices = host
        .output_devices()
        .map_err(|e| format!("cannot enumerate output devices: {e}"))?;
    let mut names = Vec::new();
    for device in devices {
        match device.name() {
            Ok(name) => names.push(name),
            Err(e) => tracing::debug!("Skipping output device with unreadable name: {e}"),
        }
    }
    Ok((names, default_name))
}

/// Find the `occurrence`-th (1-based) OS output device named
/// `hardware_name` (audio thread only). Enumeration order follows the OS
/// backend; duplicate suffixes are best-effort for this reason.
pub fn find_host_device(hardware_name: &str, occurrence: usize) -> Option<rodio::cpal::Device> {
    use rodio::cpal::traits::{DeviceTrait, HostTrait};

    let occurrence = occurrence.max(1);
    let host = rodio::cpal::default_host();
    let devices = host.output_devices().ok()?;
    let mut seen = 0usize;
    for device in devices {
        match device.name() {
            Ok(name) if name == hardware_name => {
                seen += 1;
                if seen == occurrence {
                    return Some(device);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hardware(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn default_selection_matches_case_insensitively() {
        for raw in ["default", "Default", "DEFAULT", "  default  "] {
            assert!(is_default_selection(raw), "{raw:?}");
            assert_eq!(normalize_selection(raw), "default");
        }
        assert!(!is_default_selection("Speakers"));
        assert!(!is_default_selection(""));
        assert_eq!(normalize_selection(""), "default");
        assert_eq!(normalize_selection("   "), "default");
        assert_eq!(normalize_selection("  Speakers  "), "Speakers");
    }

    #[test]
    fn list_always_exposes_logical_default_first() {
        let list = build_device_list(vec![], None, "default");
        assert_eq!(list.devices.len(), 1);
        assert_eq!(list.devices[0].id, "default");
        assert_eq!(list.devices[0].name, "System Default");
        assert!(list.devices[0].is_default);
        assert!(list.devices[0].is_selected);
        assert_eq!(list.selected, "default");
    }

    #[test]
    fn hardware_entries_carry_selection_and_default_flags() {
        let list = build_device_list(
            hardware(&["Speakers (USB)", "CABLE Input"]),
            Some("Speakers (USB)"),
            "CABLE Input",
        );
        assert_eq!(list.devices.len(), 3);
        let speakers = &list.devices[1];
        assert_eq!(speakers.id, "Speakers (USB)");
        assert_eq!(speakers.name, "Speakers (USB)");
        assert!(speakers.is_default);
        assert!(!speakers.is_selected);
        let cable = &list.devices[2];
        assert!(!cable.is_default);
        assert!(cable.is_selected);
        assert_eq!(list.selected, "CABLE Input");
    }

    #[test]
    fn duplicate_names_gain_unique_ids() {
        let list = build_device_list(
            hardware(&["USB Audio", "USB Audio", "USB Audio"]),
            None,
            "USB Audio #2",
        );
        let ids: Vec<&str> = list.devices.iter().skip(1).map(|d| d.id.as_str()).collect();
        assert_eq!(ids, ["USB Audio", "USB Audio #2", "USB Audio #3"]);
        assert!(list.devices[2].is_selected);
        assert!(!list.devices[1].is_selected);
        assert!(!list.devices[3].is_selected);
    }

    #[test]
    fn missing_selection_matches_nothing_but_is_reported() {
        let list = build_device_list(hardware(&["Speakers"]), None, "Unplugged USB");
        assert!(list.devices.iter().all(|d| !d.is_selected));
        assert_eq!(list.selected, "Unplugged USB");
    }

    #[test]
    fn match_default_requests_os_default() {
        assert_eq!(
            match_hardware("default", &hardware(&["Speakers"])),
            Some(DeviceMatch::Default)
        );
        assert_eq!(
            match_hardware("DEFAULT", &hardware(&[])),
            Some(DeviceMatch::Default)
        );
    }

    #[test]
    fn match_exact_and_duplicate_ids() {
        let names = hardware(&["Speakers", "USB Audio", "USB Audio"]);
        assert_eq!(
            match_hardware("Speakers", &names),
            Some(DeviceMatch::Named {
                name: "Speakers".to_string(),
                occurrence: 1,
            })
        );
        assert_eq!(
            match_hardware("USB Audio", &names),
            Some(DeviceMatch::Named {
                name: "USB Audio".to_string(),
                occurrence: 1,
            })
        );
        assert_eq!(
            match_hardware("USB Audio #2", &names),
            Some(DeviceMatch::Named {
                name: "USB Audio".to_string(),
                occurrence: 2,
            })
        );
        assert_eq!(match_hardware("Nope", &names), None);
        // Blank input never reaches matching: callers normalize first
        // (`normalize_selection("") == "default"`).
        assert_eq!(match_hardware("", &names), None);
        assert_eq!(match_hardware("USB Audio #3", &names), None);
    }

    #[test]
    fn device_list_serializes_for_option_sources() {
        let list = build_device_list(
            hardware(&["Speakers (USB)"]),
            Some("Speakers (USB)"),
            "default",
        );
        let json = serde_json::to_value(&list).unwrap();
        assert_eq!(json["selected"], "default");
        assert_eq!(json["devices"][0]["id"], "default");
        assert_eq!(json["devices"][0]["name"], "System Default");
        assert_eq!(json["devices"][1]["id"], "Speakers (USB)");
        assert_eq!(json["devices"][1]["name"], "Speakers (USB)");
    }

    #[test]
    fn active_device_distinguishes_configured_from_resolved() {
        let active = ActiveOutputDevice {
            device: "CABLE Input".to_string(),
            resolved_name: None,
            available: false,
        };
        let json = serde_json::to_value(&active).unwrap();
        assert_eq!(json["device"], "CABLE Input");
        assert!(json["resolved_name"].is_null());
        assert_eq!(json["available"], false);
    }
}
