import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

/// PHPickerViewController wrapper that picks one or more videos.
///
/// We don't request photo-library auth — PHPicker is privacy-mediated,
/// the user explicitly chose these assets. We don't get stable PHAsset
/// identifiers without that auth, but we generate our own asset_id
/// (user_hash + UUID) so dedup still works for the "same file picked
/// twice" case.
struct PickerView: UIViewControllerRepresentable {
    let userHash: String
    /// Maximum number of videos selectable. 0 = unlimited (Apple's spec).
    var selectionLimit: Int = 0
    /// Called once per successfully staged video. Fires on the main actor
    /// in arrival order — videos appear in UploadManager as their data
    /// becomes available.
    var onPicked: (URL, String) -> Void
    var onCancelled: () -> Void
    var onError: (String) -> Void

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .videos
        config.selectionLimit = selectionLimit
        config.preferredAssetRepresentationMode = .current
        let vc = PHPickerViewController(configuration: config)
        vc.delegate = context.coordinator
        return vc
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: PickerView
        init(parent: PickerView) { self.parent = parent }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)

            guard !results.isEmpty else {
                parent.onCancelled()
                return
            }

            let typeId = UTType.movie.identifier
            for result in results {
                guard result.itemProvider.hasItemConformingToTypeIdentifier(typeId) else {
                    DispatchQueue.main.async { self.parent.onError("Selected item isn't a video") }
                    continue
                }
                // Each loadFileRepresentation call is async; we fire them
                // in parallel and report each on the main thread as it
                // lands. The UploadManager handles dedupe + queuing so
                // out-of-order arrival is fine.
                result.itemProvider.loadFileRepresentation(forTypeIdentifier: typeId) { url, error in
                    if let error {
                        DispatchQueue.main.async { self.parent.onError(error.localizedDescription) }
                        return
                    }
                    guard let url else {
                        DispatchQueue.main.async { self.parent.onError("No file URL returned") }
                        return
                    }
                    let dest = UploadStaging.stagingURL(for: url.lastPathComponent)
                    do {
                        try FileManager.default.copyItem(at: url, to: dest)
                    } catch {
                        DispatchQueue.main.async {
                            self.parent.onError("Copy failed: \(error.localizedDescription)")
                        }
                        return
                    }
                    DispatchQueue.main.async {
                        self.parent.onPicked(dest, url.lastPathComponent)
                    }
                }
            }
        }
    }
}
