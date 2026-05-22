import Foundation
import Combine
import UIKit

/// Owns the active upload list and orchestrates the chunked-upload
/// protocol against `/api/upload/iphone/{init,part,complete}`.
///
/// PR 2 uses `URLSession.shared` (foreground only). PR 4 swaps to
/// `URLSessionConfiguration.background` so uploads survive backgrounding.
@MainActor
final class UploadManager: ObservableObject {
    static let shared = UploadManager()

    @Published private(set) var uploads: [UploadState] = []

    private let chunkSize: Int64 = 50 * 1024 * 1024  // 50 MB
    private let maxConcurrentParts = 3

    private init() {
        uploads = UploadStore.loadAll()
    }

    // MARK: - Public API

    /// Enqueue and immediately start uploading a local file.
    /// `localFileURL` MUST point to a file inside the app sandbox that
    /// will remain stable through the upload — i.e. the caller has
    /// already copied the picker's ephemeral URL into our tmp dir.
    func enqueue(
        localFileURL: URL,
        assetId: String,
        filename: String,
        userHash: String
    ) {
        let size = (try? FileManager.default.attributesOfItem(atPath: localFileURL.path)[.size] as? NSNumber)?.int64Value ?? 0
        let state = UploadState.make(
            assetId: assetId,
            filename: filename,
            sourcePath: localFileURL.path,
            totalBytes: size,
            chunkSize: chunkSize
        )
        uploads.insert(state, at: 0)
        UploadStore.save(state)
        Task { await self.run(stateId: state.id) }
    }

    func retry(id: String) {
        guard let idx = uploads.firstIndex(where: { $0.id == id }),
              uploads[idx].status == .failed else { return }
        uploads[idx].status = .queued
        uploads[idx].errorMessage = nil
        UploadStore.save(uploads[idx])
        Task { await self.run(stateId: id) }
    }

    /// Pick up an upload that was interrupted by a crash / app kill.
    /// Different from `retry`: this assumes the state is mid-flight,
    /// not a confirmed-failed one. The run() loop knows how to skip
    /// init/parts that are already on file in `partsDone`.
    func resume(id: String) {
        guard let idx = uploads.firstIndex(where: { $0.id == id }) else { return }
        let status = uploads[idx].status
        guard status != .completed && status != .failed else { return }
        Task { await self.run(stateId: id) }
    }

    func markFailed(id: String, reason: String) {
        update(stateId: id) {
            $0.status = .failed
            $0.errorMessage = reason
        }
    }

    func discard(id: String) {
        guard let idx = uploads.firstIndex(where: { $0.id == id }) else { return }
        let state = uploads[idx]
        UploadStore.delete(state)
        // Try to clean up the tmp source file.
        try? FileManager.default.removeItem(atPath: state.sourcePath)
        uploads.remove(at: idx)
    }

    // MARK: - Orchestration

    private func run(stateId: String) async {
        guard let idx0 = uploads.firstIndex(where: { $0.id == stateId }) else { return }

        // Ask iOS for extra runtime in case the user backgrounds the app.
        // ~30s on most devices; not as good as true background URLSession
        // but enough to finish small uploads or a few more chunks.
        let bgTaskName = "upload-\(stateId)"
        var bgTask = UIBackgroundTaskIdentifier.invalid
        bgTask = UIApplication.shared.beginBackgroundTask(withName: bgTaskName) {
            UIApplication.shared.endBackgroundTask(bgTask)
            bgTask = .invalid
        }
        defer {
            if bgTask != .invalid {
                UIApplication.shared.endBackgroundTask(bgTask)
            }
        }

        // 1) Init step — skip if we already have an upload_id from a
        // previous run (resume case). The Worker keeps multipart state
        // around indefinitely, so resume can pick up days later.
        if uploads[idx0].uploadId.isEmpty {
            update(stateId: stateId) { $0.status = .initializing }
            let initResult: InitResponse
            do {
                initResult = try await postInit(state: uploads[idx0])
            } catch {
                update(stateId: stateId) {
                    $0.status = .failed
                    $0.errorMessage = "init: \(error.localizedDescription)"
                }
                return
            }
            if initResult.status == "duplicate" {
                update(stateId: stateId) {
                    $0.uploadId = initResult.video_id
                    $0.status = .completed
                    $0.completedAt = Date()
                    $0.bytesUploaded = $0.totalBytes
                }
                return
            }
            update(stateId: stateId) {
                $0.uploadId = initResult.upload_id ?? initResult.video_id
                $0.status = .uploading
            }
        } else {
            // Resume — just make sure we report the right status
            update(stateId: stateId) { $0.status = .uploading }
        }

        // 2) Parts step — uploadAllParts() skips any partNumber that's
        // already in state.partsDone, so resume just continues.
        let parts: [UploadState.CompletedPart]
        do {
            parts = try await uploadAllParts(stateId: stateId)
        } catch {
            update(stateId: stateId) {
                $0.status = .failed
                $0.errorMessage = "parts: \(error.localizedDescription)"
            }
            return
        }

        // 3) Complete step
        update(stateId: stateId) { $0.status = .finalizing }
        do {
            _ = try await postComplete(stateId: stateId, parts: parts)
        } catch {
            update(stateId: stateId) {
                $0.status = .failed
                $0.errorMessage = "complete: \(error.localizedDescription)"
            }
            return
        }

        if let idx = uploads.firstIndex(where: { $0.id == stateId }) {
            try? FileManager.default.removeItem(atPath: uploads[idx].sourcePath)
        }
        update(stateId: stateId) {
            $0.status = .completed
            $0.completedAt = Date()
            $0.bytesUploaded = $0.totalBytes
        }

        // Kick off server-side status polling so the row can flip from
        // "uploaded" -> "ready in gallery" once the pipeline finishes.
        if let videoId = uploads.first(where: { $0.id == stateId })?.uploadId {
            StatusPoller.shared.start(videoId: videoId) { [weak self] resp in
                Task { @MainActor in
                    self?.update(stateId: stateId) {
                        $0.serverStatus = resp.status
                    }
                }
            }
        }
    }

    private func uploadAllParts(stateId: String) async throws -> [UploadState.CompletedPart] {
        guard let state = uploads.first(where: { $0.id == stateId }) else {
            throw UploadError.stateMissing
        }
        let totalParts = Int((state.totalBytes + state.chunkSize - 1) / state.chunkSize)
        guard totalParts > 0 else { throw UploadError.emptyFile }

        // Seed results with anything we already finished in a prior run.
        var results: [Int: UploadState.CompletedPart] = [:]
        for done in state.partsDone {
            results[done.partNumber] = done
        }
        let alreadyDone = Set(state.partsDone.map { $0.partNumber })
        let remaining = (1...totalParts).filter { !alreadyDone.contains($0) }

        if remaining.isEmpty {
            return (1...totalParts).compactMap { results[$0] }
        }

        try await withThrowingTaskGroup(of: UploadState.CompletedPart.self) { group in
            var nextIdx = 0
            var inFlight = 0

            func startOne(_ pn: Int) {
                group.addTask { [weak self] in
                    guard let self else { throw UploadError.stateMissing }
                    return try await self.uploadOnePart(stateId: stateId, partNumber: pn, totalParts: totalParts)
                }
            }

            while nextIdx < remaining.count && inFlight < maxConcurrentParts {
                startOne(remaining[nextIdx]); nextIdx += 1; inFlight += 1
            }

            while inFlight > 0 {
                guard let part = try await group.next() else { break }
                results[part.partNumber] = part
                inFlight -= 1
                if nextIdx < remaining.count {
                    startOne(remaining[nextIdx]); nextIdx += 1; inFlight += 1
                }
            }
        }

        return (1...totalParts).compactMap { results[$0] }
    }

    private func uploadOnePart(stateId: String, partNumber: Int, totalParts: Int) async throws -> UploadState.CompletedPart {
        guard let state = uploads.first(where: { $0.id == stateId }) else {
            throw UploadError.stateMissing
        }
        let url = URL(fileURLWithPath: state.sourcePath)
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let offset = Int64(partNumber - 1) * state.chunkSize
        try handle.seek(toOffset: UInt64(offset))
        let want = Int(min(state.chunkSize, state.totalBytes - offset))
        let chunk = try handle.read(upToCount: want) ?? Data()

        var req = URLRequest(url: APIClient.baseURL
            .appendingPathComponent("api/upload/iphone/\(state.uploadId)/\(partNumber)"))
        req.httpMethod = "PUT"
        req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        if let jwt = TokenStore.load() {
            req.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
        }

        // Retry up to 3 times on transient errors
        var lastError: Error?
        for attempt in 0..<3 {
            do {
                let (data, response) = try await URLSession.shared.upload(for: req, from: chunk)
                guard let http = response as? HTTPURLResponse else {
                    throw UploadError.badResponse
                }
                if !(200..<300).contains(http.statusCode) {
                    let body = String(data: data, encoding: .utf8) ?? ""
                    throw UploadError.http(status: http.statusCode, body: body)
                }
                let resp = try JSONDecoder().decode(PartResponse.self, from: data)
                await MainActor.run { [self] in
                    self.update(stateId: stateId) {
                        $0.partsDone.append(.init(partNumber: resp.partNumber, etag: resp.etag, size: Int64(chunk.count)))
                        $0.bytesUploaded += Int64(chunk.count)
                    }
                }
                return .init(partNumber: resp.partNumber, etag: resp.etag, size: Int64(chunk.count))
            } catch {
                lastError = error
                // Exponential backoff: 1s, 3s
                if attempt < 2 {
                    try? await Task.sleep(nanoseconds: UInt64(pow(3.0, Double(attempt))) * 1_000_000_000)
                }
            }
        }
        throw lastError ?? UploadError.unknown
    }

    private func postInit(state: UploadState) async throws -> InitResponse {
        let body = InitRequest(
            asset_id: state.assetId,
            filename: state.filename,
            created_at: state.createdAtISO
        )
        var req = URLRequest(url: APIClient.baseURL.appendingPathComponent("api/upload/iphone/init"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let jwt = TokenStore.load() {
            req.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
        }
        req.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse else { throw UploadError.badResponse }
        // 200 = new upload, 409 = duplicate (treated as success by caller)
        if http.statusCode == 409 {
            return try JSONDecoder().decode(InitResponse.self, from: data)
        }
        if !(200..<300).contains(http.statusCode) {
            let bodyText = String(data: data, encoding: .utf8) ?? ""
            throw UploadError.http(status: http.statusCode, body: bodyText)
        }
        return try JSONDecoder().decode(InitResponse.self, from: data)
    }

    private func postComplete(stateId: String, parts: [UploadState.CompletedPart]) async throws -> CompleteResponse {
        guard let state = uploads.first(where: { $0.id == stateId }) else {
            throw UploadError.stateMissing
        }
        let body = CompleteRequest(parts: parts.map { .init(partNumber: $0.partNumber, etag: $0.etag) })
        var req = URLRequest(url: APIClient.baseURL
            .appendingPathComponent("api/upload/iphone/\(state.uploadId)/complete"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let jwt = TokenStore.load() {
            req.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
        }
        req.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse else { throw UploadError.badResponse }
        if !(200..<300).contains(http.statusCode) {
            let bodyText = String(data: data, encoding: .utf8) ?? ""
            throw UploadError.http(status: http.statusCode, body: bodyText)
        }
        return try JSONDecoder().decode(CompleteResponse.self, from: data)
    }

    // MARK: - State updates

    private func update(stateId: String, _ block: (inout UploadState) -> Void) {
        guard let idx = uploads.firstIndex(where: { $0.id == stateId }) else { return }
        var s = uploads[idx]
        block(&s)
        uploads[idx] = s
        UploadStore.save(s)
    }
}

// MARK: - Wire types

private struct InitRequest: Encodable {
    let asset_id: String
    let filename: String
    let created_at: String
}

private struct InitResponse: Decodable {
    let video_id: String
    let upload_id: String?  // present on 200, absent on 409
    let r2_key: String
    let asset_id: String
    let status: String?     // "duplicate" on 409
}

private struct PartResponse: Decodable {
    let partNumber: Int
    let etag: String
}

private struct CompleteRequest: Encodable {
    let parts: [PartRef]
    struct PartRef: Encodable {
        let partNumber: Int
        let etag: String
    }
}

private struct CompleteResponse: Decodable {
    let video_id: String
    let status: String
    let r2_key: String
}

enum UploadError: LocalizedError {
    case stateMissing
    case emptyFile
    case badResponse
    case http(status: Int, body: String)
    case unknown

    var errorDescription: String? {
        switch self {
        case .stateMissing: return "Upload state missing"
        case .emptyFile: return "File is empty"
        case .badResponse: return "Bad response"
        case .http(let s, let b): return "HTTP \(s): \(b)"
        case .unknown: return "Unknown error"
        }
    }
}

// MARK: - tmp dir helper (used by PickerView / RecordView to stage source files)

enum UploadStaging {
    static var directory: URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("uploads", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func stagingURL(for filename: String) -> URL {
        let safe = filename.replacingOccurrences(of: "/", with: "_")
        return directory.appendingPathComponent("\(UUID().uuidString)_\(safe)")
    }
}
