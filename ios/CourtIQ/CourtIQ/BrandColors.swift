import SwiftUI

/// Tennis Uploader brand palette — charcoal + neon yellow-green
/// (Wimbledon-tennis-ball #C7FF00). All views should pull from this
/// instead of using raw hex/RGB so palette changes stay localized.
extension Color {
    /// Pure neon accent — used for primary CTAs, active states, tab tint.
    /// Mirrors the AccentColor.colorset asset.
    static let brandAccent = Color(red: 0xC7/255, green: 0xFF/255, blue: 0x00/255)

    /// Near-black background.
    static let brandBackground = Color(red: 0x0A/255, green: 0x0A/255, blue: 0x0B/255)

    /// Card / surface color sitting on top of the background.
    static let brandSurface = Color(red: 0x1C/255, green: 0x1C/255, blue: 0x1E/255)

    /// Secondary surface (e.g. nested elements inside a card).
    static let brandSurfaceElevated = Color(red: 0x2C/255, green: 0x2C/255, blue: 0x2E/255)

    /// Primary text — off-white, not pure white.
    static let brandText = Color(red: 0xF5/255, green: 0xF5/255, blue: 0xF7/255)

    /// Muted secondary text.
    static let brandTextSecondary = Color(red: 0x9A/255, green: 0xA0/255, blue: 0xA6/255)
}
