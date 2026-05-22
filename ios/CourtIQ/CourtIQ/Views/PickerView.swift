import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

/// PHPickerViewController wrapper that picks a single video and hands
/// us a local file URL we can hand to `UploadManager`.
///
/// We don't request photo-library auth — PHPicker is privacy-mediated,
/// the user explicitly chose this asset, that's enough. We don't get a
/// stable PHAsset identifier without that auth, but we generate our
/// own asset_id (user_hash + UUID) so dedup still works for the
/// "same file picked twice" case.
struct PickerView: UIViewControllerRepresentable {
    let userHash: String
    /// Called with a staged local file URL + original filename.
    /// Called on the main actor.
    var onPicked: (URL, String) -> Void
    var onCancelled: () -> Void
    var onError: (String) -> Void

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .videos
        config.selectionLimit = 1
        config.preferredAssetRepresentationMode = .current  // original quality
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

            guard let result = results.first else {
                parent.onCancelled()
                return
            }

            let typeId = UTType.movie.identifier
            guard result.itemProvider.hasItemConformingToTypeIdentifier(typeId) else {
                parent.onError("Selected item isn't a video")
                return
            }

            // loadFileRepresentation gives us a temporary URL that we MUST
            // copy out of before the closure returns — iOS reclaims it.
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
                    DispatchQueue.main.async { self.parent.onError("Copy failed: \(error.localizedDescription)") }
                    return
                }
                DispatchQueue.main.async {
                    self.parent.onPicked(dest, url.lastPathComponent)
                }
            }
        }
    }
}
