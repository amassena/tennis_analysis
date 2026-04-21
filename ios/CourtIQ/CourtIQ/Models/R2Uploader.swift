import Foundation
import UIKit
import Photos

class R2Uploader: ObservableObject {
    @Published var isUploading = false
    @Published var progress: Double = 0
    @Published var statusMessage = ""
    @Published var completed = false

    func uploadSession(videoURL: URL, completion: @escaping (Bool) -> Void) {
        DispatchQueue.main.async {
            self.isUploading = true
            self.progress = 0.2
            self.statusMessage = "Saving to Photos..."
        }

        // Save to Photos → iCloud syncs → watcher picks up → GPU processes
        // This is the bridge until direct R2 upload is built
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { [weak self] status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    self?.statusMessage = "Photo library access denied"
                    self?.isUploading = false
                }
                completion(false)
                return
            }

            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: videoURL)
            }) { success, error in
                DispatchQueue.main.async {
                    if success {
                        self?.progress = 1.0
                        self?.statusMessage = "Saved to Photos! Your video will be automatically processed and appear on the main screen with coaching analysis."
                        self?.completed = true
                    } else {
                        self?.statusMessage = "Save failed: \(error?.localizedDescription ?? "unknown")"
                    }
                    self?.isUploading = false
                }
                completion(success)
            }
        }
    }
}
