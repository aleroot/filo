#pragma once

#include <algorithm>
#include <string>
#include <string_view>

namespace tui {

/// Tracks the active query key and selected suggestion index for autocomplete
/// pickers (command picker, mention picker).  Automatically resets the
/// selection when the query key changes.
///
/// Supports "suppress" mode so the user can dismiss a picker with ESC and
/// continue navigating input history: the picker stays hidden until the query
/// text actually changes, then re-appears as normal.
struct PickerState {
    std::string key;
    int         selected   = 0;
    bool        suppressed = false;  ///< True after user dismissed via ESC.

    /// Called on every render/event with the current query key and suggestion
    /// count.  Un-suppresses automatically when the query changes.
    void sync(std::string_view new_key, int count) {
        const std::string new_key_str(new_key);
        if (new_key_str != key) {
            key        = new_key_str;
            selected   = 0;
            suppressed = false;  // New query → show the picker again.
        }
        selected = count > 0 ? std::clamp(selected, 0, count - 1) : 0;
    }

    void navigate_down(int count) {
        if (count > 0) { selected = (selected + 1) % count; }
    }

    void navigate_up(int count) {
        if (count > 0) { selected = (selected + count - 1) % count; }
    }

    /// Full reset — clears key, selection, and suppression (e.g., after a
    /// completion is accepted and the input text is replaced).
    void clear() {
        key.clear();
        selected   = 0;
        suppressed = false;
    }

    /// Dismiss the picker without changing the key.  The picker stays hidden
    /// until the query text changes.  Used for ESC so the user can continue
    /// navigating history even when the loaded entry happens to be a command.
    void suppress() {
        selected   = 0;
        suppressed = true;
    }
};

}  // namespace tui
