import Foundation
import Combine

/// Cross-tab navigation state. Lets the Upload tab tell the Gallery tab
/// "after the user taps View, scroll to anchor #<vid>" without a manual
/// hand-off through View hierarchies.
@MainActor
final class AppNavigation: ObservableObject {
    enum Tab: Hashable {
        case upload
        case gallery
    }

    @Published var selectedTab: Tab = .upload
    /// Set this then switch to .gallery; GalleryTabView consumes it and
    /// nils it out so subsequent reloads don't re-jump.
    @Published var pendingGalleryAnchor: String?

    /// Convenience for "go look at this video in the gallery".
    func openGallery(anchor videoId: String? = nil) {
        pendingGalleryAnchor = videoId
        selectedTab = .gallery
    }
}
