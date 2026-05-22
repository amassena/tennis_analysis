import Foundation

/// Per-upload state, persisted to Application Support so PR 4's
/// `UploadResumer` can recover after a crash. PR 2 only writes the file
/// on terminal events (success/failure); PR 4 will write on every part
/// completion so resume can pick up mid-upload.
struct UploadState: Codable, Identifiable, Equatable {
    /// Server-issued upload id (we use this as `id` for the SwiftUI list).
    var uploadId: String
    /// Stable asset id sent to the Worker (used to compute video_id).
    let assetId: String
    let filename: String
    /// Local file backing this upload (in app tmp dir).
    let sourcePath: String
    let createdAtISO: String
    let totalBytes: Int64
    let chunkSize: Int64

    var partsDone: [CompletedPart]
    var status: Status
    var bytesUploaded: Int64
    var errorMessage: String?
    var startedAt: Date
    var completedAt: Date?

    var id: String { uploadId.isEmpty ? assetId : uploadId }

    var progress: Double {
        guard totalBytes > 0 else { return 0 }
        return min(1.0, Double(bytesUploaded) / Double(totalBytes))
    }

    enum Status: String, Codable {
        case queued       // created but not yet started
        case initializing // POST /api/upload/iphone/init in flight
        case uploading    // PUTting parts
        case finalizing   // POST /complete in flight
        case completed
        case failed
    }

    struct CompletedPart: Codable, Equatable {
        let partNumber: Int
        let etag: String
        let size: Int64
    }

    static func make(
        assetId: String,
        filename: String,
        sourcePath: String,
        totalBytes: Int64,
        chunkSize: Int64
    ) -> UploadState {
        UploadState(
            uploadId: "",
            assetId: assetId,
            filename: filename,
            sourcePath: sourcePath,
            createdAtISO: ISO8601DateFormatter().string(from: Date()),
            totalBytes: totalBytes,
            chunkSize: chunkSize,
            partsDone: [],
            status: .queued,
            bytesUploaded: 0,
            errorMessage: nil,
            startedAt: Date(),
            completedAt: nil
        )
    }
}

/// Disk persistence for UploadState records.
///
/// Layout:
///   Library/Application Support/QuickUpload/state_<id>.json
enum UploadStore {
    static var directory: URL {
        let fm = FileManager.default
        let base = try! fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let dir = base.appendingPathComponent("QuickUpload", isDirectory: true)
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func loadAll() -> [UploadState] {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
            return []
        }
        var states: [UploadState] = []
        let decoder = JSONDecoder()
        for url in files where url.lastPathComponent.hasPrefix("state_") && url.pathExtension == "json" {
            if let data = try? Data(contentsOf: url),
               let state = try? decoder.decode(UploadState.self, from: data) {
                states.append(state)
            }
        }
        return states.sorted { $0.startedAt > $1.startedAt }
    }

    static func save(_ state: UploadState) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(state) else { return }
        let url = directory.appendingPathComponent("state_\(state.id).json")
        try? data.write(to: url, options: .atomic)
    }

    static func delete(_ state: UploadState) {
        let url = directory.appendingPathComponent("state_\(state.id).json")
        try? FileManager.default.removeItem(at: url)
    }
}
