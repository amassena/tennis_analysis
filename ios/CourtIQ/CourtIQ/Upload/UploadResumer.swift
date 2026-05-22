import Foundation

/// Scans persisted upload state at launch and resumes any in-flight
/// upload that didn't complete cleanly.
///
/// Resume strategy:
///   - .completed   → leave alone (history row)
///   - .failed      → leave alone (user must tap retry)
///   - .queued      → kick off via UploadManager.run()
///   - .initializing/.uploading/.finalizing → resume mid-flight
///
/// The actual resume work is delegated to UploadManager which knows
/// the protocol; this type just decides what to do per state.
@MainActor
enum UploadResumer {
    static func resumeOnLaunch() {
        let states = UploadStore.loadAll()
        for state in states {
            switch state.status {
            case .completed, .failed:
                continue
            case .queued, .initializing, .uploading, .finalizing:
                // Verify the source file still exists. If it's gone
                // (user wiped tmp dir manually, OS cleaned up, etc.),
                // mark as failed.
                if !FileManager.default.fileExists(atPath: state.sourcePath) {
                    UploadManager.shared.markFailed(id: state.id, reason: "Source file no longer available")
                } else {
                    UploadManager.shared.resume(id: state.id)
                }
            }
        }
    }
}
