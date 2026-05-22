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
        // First pass: dedupe stale state files left over from the
        // pre-fix bug where state.id switched mid-upload (asset_id ->
        // upload_id) and UploadStore wrote a second `state_<upload_id>.json`
        // without deleting the original. Group by sourcePath; if two
        // states share the same backing file, keep the most recently
        // touched and discard the rest.
        dedupeStaleStateFiles()

        let states = UploadStore.loadAll()
        for state in states {
            switch state.status {
            case .completed, .failed:
                continue
            case .queued, .initializing, .uploading, .finalizing:
                if !FileManager.default.fileExists(atPath: state.sourcePath) {
                    UploadManager.shared.markFailed(id: state.id, reason: "Source file no longer available")
                } else {
                    UploadManager.shared.resume(id: state.id)
                }
            }
        }
    }

    private static func dedupeStaleStateFiles() {
        let all = UploadStore.loadAll()
        var bestBySource: [String: UploadState] = [:]
        for s in all {
            if let existing = bestBySource[s.sourcePath] {
                // Prefer .completed > anything else; otherwise newest startedAt.
                let pickNew: Bool
                if s.status == .completed && existing.status != .completed {
                    pickNew = true
                } else if existing.status == .completed && s.status != .completed {
                    pickNew = false
                } else {
                    pickNew = s.startedAt > existing.startedAt
                }
                if pickNew {
                    UploadStore.delete(existing)
                    bestBySource[s.sourcePath] = s
                } else {
                    UploadStore.delete(s)
                }
            } else {
                bestBySource[s.sourcePath] = s
            }
        }
    }
}
