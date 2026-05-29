import Foundation
import Combine
import UIKit
import AVFoundation
import CryptoKit

/// Owns the active upload list and orchestrates the chunked-upload
/// protocol against `/api/upload/iphone/{init,part,complete}`.
///
/// PR 2 uses `URLSession.shared` (foreground only). PR 4 swaps to
/// `URLSessionConfiguration.background` so uploads survive backgrounding.
@MainActor
final class UploadManager: ObservableObject {
    static let shared = UploadManager()

    @Published private(set) var uploads: [UploadState] = [] {
        didSet { updateScreenWake() }
    }

    /// Keep the screen awake while an upload is actively in flight. iOS
    /// auto-lock suspends the app (and the foreground URLSession), pausing
    /// the upload — for the patient/foreground approach we prevent the
    /// screen from sleeping so a long upload keeps progressing. Restored
    /// the moment no upload is active.
    private func updateScreenWake() {
        let active = uploads.contains {
            $0.status == .queued || $0.status == .initializing
                || $0.status == .uploading || $0.status == .finalizing
        }
        UIApplication.shared.isIdleTimerDisabled = active
    }

    private let chunkSize: Int64 = 50 * 1024 * 1024  // 50 MB
    private let maxConcurrentParts = 3

    /// User-controlled toggle, persisted via UserDefaults.
    /// Enforced by each session's `allowsCellularAccess`.
    @Published var wifiOnly: Bool {
        didSet {
            UserDefaults.standard.set(wifiOnly, forKey: "quickUpload_wifiOnly")
            // Rebuild the control session immediately. The background session
            // can't be swapped live (one per identifier per process), so it
            // adopts the new policy on next launch — acceptable for a rarely
            // flipped setting.
            controlSessionCache = nil
        }
    }

    /// stateIds with a run() loop currently executing. Guards against
    /// double-running the same upload when both launch-resume and
    /// foreground-resume (or a manual retry) fire for it.
    private var running = Set<String>()

    /// Foreground session for the small control calls (init + complete).
    /// Rebuilt when the WiFi-only toggle flips.
    private var controlSessionCache: URLSession?
    var controlSession: URLSession {
        if let s = controlSessionCache { return s }
        let cfg = URLSessionConfiguration.default
        cfg.allowsCellularAccess = !wifiOnly
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 60 * 60
        let s = URLSession(configuration: cfg)
        controlSessionCache = s
        return s
    }

    /// Background session for the heavy part uploads. Survives app
    /// suspension/termination so a multi-GB upload keeps going while the
    /// app is backgrounded or the screen is locked (#21). Delegate-driven:
    /// parts finish via BackgroundUploadDelegate → handlePartTaskCompletion.
    /// Built ONCE per process (you can't have two background sessions with
    /// the same identifier).
    static let backgroundSessionId = "com.amassena.courtiq.CourtIQ.upload"
    private let bgDelegate = BackgroundUploadDelegate()
    private(set) lazy var uploadSession: URLSession = {
        let cfg = URLSessionConfiguration.background(withIdentifier: Self.backgroundSessionId)
        cfg.allowsCellularAccess = !wifiOnly
        cfg.sessionSendsLaunchEvents = true       // relaunch the app for completion events
        cfg.isDiscretionary = false               // start promptly, don't wait for "ideal" conditions
        cfg.timeoutIntervalForResource = 7 * 24 * 3600  // a week to finish a huge upload
        return URLSession(configuration: cfg, delegate: bgDelegate, delegateQueue: nil)
    }()

    /// "stateId#partNumber" for parts with a background task in flight —
    /// stops scheduleParts from double-scheduling the same part.
    private var inFlightParts = Set<String>()
    /// Transient-failure retry counts, keyed "stateId#partNumber".
    private var partRetries: [String: Int] = [:]
    /// stateIds with a /complete call in flight — prevents a double-finalize
    /// if a foreground-resume fires during the finalize window.
    private var finalizing = Set<String>()

    /// Stored by AppDelegate.handleEventsForBackgroundURLSession; called once
    /// the background session has delivered all queued completion callbacks.
    var backgroundEventsCompletion: (() -> Void)?
    func finishBackgroundEvents() {
        let h = backgroundEventsCompletion
        backgroundEventsCompletion = nil
        h?()
    }

    private init() {
        self.wifiOnly = UserDefaults.standard.object(forKey: "quickUpload_wifiOnly") as? Bool ?? true
        uploads = UploadStore.loadAll()
    }

    // MARK: - Public API

    /// Enqueue and immediately start uploading a local file.
    /// `localFileURL` MUST point to a file inside the app sandbox that
    /// will remain stable through the upload — i.e. the caller has
    /// already copied the picker's ephemeral URL into our tmp dir.
    func enqueue(
        localFileURL: URL,
        filename: String,
        userHash: String
    ) {
        let size = (try? FileManager.default.attributesOfItem(atPath: localFileURL.path)[.size] as? NSNumber)?.int64Value ?? 0
        // Extract the real recording timestamp from the video's
        // metadata — falls back to "now" if unavailable so we never
        // block the upload. The server pipeline also re-extracts the
        // date from the source MOV; this just makes the date correct
        // for the in-app surfaces BEFORE processing finishes.
        let rawCreation = videoCreationDate(at: localFileURL)
        let recordedAt = rawCreation ?? Date()

        // Deterministic, content-derived asset_id. The same library video
        // re-picked must map to the SAME id so the server dedups/resumes
        // instead of creating a duplicate upload (issue #27). PHPicker is
        // permission-free, so we can't use a PHAsset identifier — but
        // filename + byte-size + creation-date is a stable signature for a
        // given video without any photo-library auth. (Note: do NOT fold in
        // the `recordedAt` fallback `Date()`, which is non-deterministic.)
        let assetId = Self.stableAssetId(
            userHash: userHash, filename: filename, size: size, creation: rawCreation)

        // Client-side dedup: if this exact video is already known, don't add
        // a second row (which would also collide on UploadStore's state file
        // and confuse the SwiftUI list by id). Resume a failed one; leave an
        // active or completed one alone (the server confirms duplicates on
        // re-init anyway).
        if let existing = uploads.first(where: { $0.id == assetId }) {
            switch existing.status {
            case .failed:
                retry(id: assetId)
            case .queued, .initializing, .uploading, .finalizing, .completed:
                break
            }
            return
        }

        let state = UploadState.make(
            assetId: assetId,
            filename: filename,
            sourcePath: localFileURL.path,
            totalBytes: size,
            chunkSize: chunkSize,
            recordedAt: recordedAt
        )
        uploads.insert(state, at: 0)
        UploadStore.save(state)
        Task { await self.run(stateId: state.id) }
    }

    /// SHA256(userHash | filename | size | creation-epoch) truncated — a
    /// stable id for a given video so re-picks dedup. See `enqueue`.
    private static func stableAssetId(
        userHash: String, filename: String, size: Int64, creation: Date?
    ) -> String {
        let creationStamp = creation.map { String(Int($0.timeIntervalSince1970)) } ?? ""
        let seed = "\(userHash)|\(filename)|\(size)|\(creationStamp)"
        let hex = SHA256.hash(data: Data(seed.utf8))
            .map { String(format: "%02x", $0) }.joined().prefix(16)
        return "\(userHash)_\(hex)"
    }

    /// Reads `creationDate` from the AVURLAsset metadata. iPhone Camera
    /// embeds it in the .mov/.mp4 (com.apple.quicktime.creationdate);
    /// returns nil for files without it.
    private func videoCreationDate(at url: URL) -> Date? {
        let asset = AVURLAsset(url: url)
        if let d = asset.creationDate?.dateValue { return d }
        for fmt in asset.availableMetadataFormats {
            for item in asset.metadata(forFormat: fmt) {
                if item.commonKey == .commonKeyCreationDate,
                   let d = item.dateValue {
                    return d
                }
            }
        }
        return nil
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

    /// A connectivity / app-suspension error that should NOT terminally
    /// fail the upload — it stays resumable so launch/foreground resume
    /// retries from the next missing part. Permanent errors (auth, bad
    /// request) still fail so the user sees them.
    private func isTransient(_ error: Error) -> Bool {
        if let u = error as? URLError {
            switch u.code {
            case .notConnectedToInternet, .networkConnectionLost, .timedOut,
                 .cannotConnectToHost, .cancelled, .dataNotAllowed,
                 .internationalRoamingOff, .callIsActive:
                return true
            default:
                return false
            }
        }
        return false
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
        // Don't double-run: launch-resume, foreground-resume, and manual
        // retry can all target the same upload. Only one loop at a time.
        guard !running.contains(stateId) else { return }
        running.insert(stateId)
        defer { running.remove(stateId) }

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
                // Already on the server — drop the staged source so it
                // doesn't sit in Application Support forever.
                if let s = uploads.first(where: { $0.id == stateId }) {
                    try? FileManager.default.removeItem(atPath: s.sourcePath)
                }
                update(stateId: stateId) {
                    $0.uploadId = initResult.video_id
                    $0.status = .completed
                    $0.completedAt = Date()
                    $0.bytesUploaded = $0.totalBytes
                }
                return
            }
            if initResult.status == "resume" {
                // Server already has an in-flight multipart for this asset.
                // Adopt its upload_id and seed partsDone from the server's
                // record so uploadAllParts skips what already landed — this is
                // what makes re-picking the same video after a stall / app
                // kill / REINSTALL continue instead of restarting from 0.
                let serverParts = initResult.parts ?? []
                update(stateId: stateId) {
                    $0.uploadId = initResult.upload_id ?? initResult.video_id
                    $0.partsDone = serverParts.map { p in
                        UploadState.CompletedPart(partNumber: p.partNumber, etag: p.etag, size: p.size)
                    }
                    $0.bytesUploaded = serverParts.reduce(Int64(0)) { acc, p in acc + p.size }
                    $0.status = .uploading
                }
            } else {
                update(stateId: stateId) {
                    $0.uploadId = initResult.upload_id ?? initResult.video_id
                    $0.status = .uploading
                }
            }
        } else {
            // Resume — just make sure we report the right status
            update(stateId: stateId) { $0.status = .uploading }
        }

        // 2) Parts — hand off to the BACKGROUND session and return. Each part
        // uploads as a background task; the delegate calls
        // handlePartTaskCompletion(), which refills the concurrency window and
        // triggers finalize() once the whole file is up. This is what lets a
        // multi-GB upload keep running while the app is backgrounded or the
        // screen is locked (#21).
        scheduleParts(stateId: stateId)
    }

    // MARK: - Background part scheduling

    private func totalParts(_ state: UploadState) -> Int {
        Int((state.totalBytes + state.chunkSize - 1) / state.chunkSize)
    }

    /// Schedule up to `maxConcurrentParts` not-yet-uploaded, not-in-flight
    /// parts as background upload tasks. Idempotent — safe to call from run(),
    /// from each part completion, on resume, and on launch reconnect.
    private func scheduleParts(stateId: String) {
        guard let state = uploads.first(where: { $0.id == stateId }) else { return }
        guard state.status == .uploading, !state.uploadId.isEmpty else { return }
        let total = totalParts(state)
        guard total > 0 else { markFailed(id: stateId, reason: "File is empty"); return }

        let done = Set(state.partsDone.map { $0.partNumber })
        let inFlightForState = inFlightParts.filter { $0.hasPrefix("\(stateId)#") }.count

        // Whole file up → finalize once nothing is still in flight.
        if done.count >= total {
            if inFlightForState == 0 { finalize(stateId: stateId) }
            return
        }

        var slots = maxConcurrentParts - inFlightForState
        guard slots > 0 else { return }
        for pn in 1...total where slots > 0 {
            if done.contains(pn) { continue }
            let key = "\(stateId)#\(pn)"
            if inFlightParts.contains(key) { continue }
            if enqueuePart(state: state, partNumber: pn) {
                inFlightParts.insert(key)
                slots -= 1
            }
        }
    }

    /// Stage part `partNumber`'s chunk to a temp file and start a background
    /// upload task for it (background sessions require a file body, not Data).
    /// Returns false if the chunk couldn't be staged.
    private func enqueuePart(state: UploadState, partNumber: Int) -> Bool {
        let offset = Int64(partNumber - 1) * state.chunkSize
        let want = Int(min(state.chunkSize, state.totalBytes - offset))
        guard want > 0 else { return false }

        let tempURL = Self.partTempURL(stateId: state.id, partNumber: partNumber)
        do {
            let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: state.sourcePath))
            defer { try? handle.close() }
            try handle.seek(toOffset: UInt64(offset))
            let chunk = try handle.read(upToCount: want) ?? Data()
            try chunk.write(to: tempURL, options: .atomic)
        } catch {
            return false
        }

        var req = URLRequest(url: APIClient.baseURL
            .appendingPathComponent("api/upload/iphone/\(state.uploadId)/\(partNumber)"))
        req.httpMethod = "PUT"
        req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        if let jwt = TokenStore.load() {
            req.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
        }
        let task = uploadSession.uploadTask(with: req, fromFile: tempURL)
        task.taskDescription = UploadTaskTag(
            stateId: state.id, partNumber: partNumber, size: Int64(want)).encoded()
        task.resume()
        return true
    }

    /// Called by BackgroundUploadDelegate when a part task finishes.
    /// `statusCode` is 0 on transport error. We DON'T need the response etag —
    /// the server records each part's etag itself (server-authoritative
    /// complete), so a 2xx is all that matters here.
    func handlePartTaskCompletion(tag: UploadTaskTag, statusCode: Int, errored: Bool) {
        let key = "\(tag.stateId)#\(tag.partNumber)"
        inFlightParts.remove(key)
        try? FileManager.default.removeItem(
            at: Self.partTempURL(stateId: tag.stateId, partNumber: tag.partNumber))
        guard uploads.contains(where: { $0.id == tag.stateId }) else { return }

        let success = !errored && (200..<300).contains(statusCode)
        if success {
            partRetries[key] = nil
            update(stateId: tag.stateId) {
                if !$0.partsDone.contains(where: { $0.partNumber == tag.partNumber }) {
                    $0.partsDone.append(.init(partNumber: tag.partNumber, etag: "", size: tag.size))
                    $0.bytesUploaded += tag.size
                }
            }
            scheduleParts(stateId: tag.stateId)
            return
        }

        // Auth / bad-request style codes are terminal; transport blips and
        // 5xx are retried a few times, then left for launch/foreground resume.
        let terminal = [400, 401, 403, 404, 413].contains(statusCode)
        if terminal {
            markFailed(id: tag.stateId, reason: "part \(tag.partNumber): HTTP \(statusCode)")
            return
        }
        let count = (partRetries[key] ?? 0) + 1
        partRetries[key] = count
        if count <= 5 {
            scheduleParts(stateId: tag.stateId)   // re-pick this part
        } else {
            partRetries[key] = nil                // back off; resume retries later
        }
    }

    /// All parts uploaded — POST /complete (server validates the whole file),
    /// then mark done and clean up. Uses the foreground control session,
    /// wrapped in a background-task assertion so a finalize landing while the
    /// app is briefly backgrounded still finishes.
    private func finalize(stateId: String) {
        guard !finalizing.contains(stateId) else { return }
        guard let state = uploads.first(where: { $0.id == stateId }),
              state.status != .completed, state.status != .failed else { return }
        finalizing.insert(stateId)
        update(stateId: stateId) { $0.status = .finalizing }
        let parts = state.partsDone
        Task { @MainActor in
            defer { finalizing.remove(stateId) }
            var bg = UIApplication.shared.beginBackgroundTask(withName: "finalize-\(stateId)")
            defer { if bg != .invalid { UIApplication.shared.endBackgroundTask(bg); bg = .invalid } }
            do {
                _ = try await postComplete(stateId: stateId, parts: parts)
            } catch {
                if case UploadError.http(let status, _) = error, status == 409 {
                    // Server says incomplete — keep uploading; the next run's
                    // init-resume reconciles partsDone with the server's list.
                    update(stateId: stateId) { $0.status = .uploading; $0.errorMessage = nil }
                    scheduleParts(stateId: stateId)
                } else if isTransient(error) {
                    update(stateId: stateId) { $0.status = .uploading; $0.errorMessage = nil }
                } else {
                    update(stateId: stateId) {
                        $0.status = .failed
                        $0.errorMessage = "complete: \(error.localizedDescription)"
                    }
                }
                return
            }
            if let s = uploads.first(where: { $0.id == stateId }) {
                try? FileManager.default.removeItem(atPath: s.sourcePath)
            }
            update(stateId: stateId) {
                $0.status = .completed
                $0.completedAt = Date()
                $0.bytesUploaded = $0.totalBytes
            }
            if let videoId = uploads.first(where: { $0.id == stateId })?.uploadId {
                StatusPoller.shared.start(videoId: videoId) { [weak self] resp in
                    Task { @MainActor in
                        self?.update(stateId: stateId) { $0.serverStatus = resp.status }
                    }
                }
            }
        }
    }

    /// On launch, re-attach to the background session: rebuild the in-flight
    /// set from any tasks still running (so we don't double-schedule), then
    /// resume. Tasks that finished while we were dead already had their
    /// completion replayed to the delegate by the system.
    func reconnectAndResume() {
        uploadSession.getAllTasks { tasks in
            Task { @MainActor in
                for t in tasks {
                    if let tag = UploadTaskTag.decode(t.taskDescription) {
                        self.inFlightParts.insert("\(tag.stateId)#\(tag.partNumber)")
                    }
                }
                UploadResumer.resumeOnLaunch()
            }
        }
    }

    /// Temp file backing a single in-flight part upload.
    static func partTempURL(stateId: String, partNumber: Int) -> URL {
        let safe = stateId.replacingOccurrences(of: "/", with: "_")
        return UploadStaging.partsDirectory.appendingPathComponent("\(safe)_\(partNumber).part")
    }

    private func postInit(state: UploadState) async throws -> InitResponse {
        let body = InitRequest(
            asset_id: state.assetId,
            filename: state.filename,
            created_at: state.createdAtISO,
            total_bytes: state.totalBytes
        )
        var req = URLRequest(url: APIClient.baseURL.appendingPathComponent("api/upload/iphone/init"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let jwt = TokenStore.load() {
            req.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
        }
        req.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await self.controlSession.data(for: req)
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
        let (data, response) = try await self.controlSession.data(for: req)
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
    let total_bytes: Int64   // lets the server validate whole-file at complete
}

private struct InitResponse: Decodable {
    let video_id: String
    let upload_id: String?  // present on 200, absent on 409
    let r2_key: String
    let asset_id: String
    let status: String?      // "duplicate" (409) | "resume" (200, in-flight exists)
    let parts: [ServerPart]? // present on "resume": parts already uploaded
}

/// A part the server already has — returned in an init "resume" response so
/// the client can skip re-uploading it (server-authoritative, survives a
/// reinstall that wiped local UploadState).
private struct ServerPart: Decodable {
    let partNumber: Int
    let etag: String
    let size: Int64
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
    case badResponse
    case http(status: Int, body: String)
    case unknown

    var errorDescription: String? {
        switch self {
        case .stateMissing: return "Upload state missing"
        case .badResponse: return "Bad response"
        case .http(let s, let b): return "HTTP \(s): \(b)"
        case .unknown: return "Unknown error"
        }
    }
}

// MARK: - tmp dir helper (used by PickerView / RecordView to stage source files)

enum UploadStaging {
    // Staged source videos MUST live in Application Support, NOT the temp
    // dir. iOS purges temporaryDirectory aggressively (on relaunch, memory
    // pressure, low storage), which for a multi-GB pick that's still
    // uploading caused "Source file no longer available" — the file got
    // wiped mid-upload or before a resume. Application Support is persistent
    // and excluded from that cleanup; we delete staged files ourselves on
    // upload completion / discard.
    static var directory: URL {
        let base = try! FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = base.appendingPathComponent("QuickUpload/sources", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        // Exclude from iCloud/iTunes backup (these are transient large
        // files we can re-pick; no need to back them up).
        var d = dir
        var rv = URLResourceValues(); rv.isExcludedFromBackup = true
        try? d.setResourceValues(rv)
        return dir
    }

    static func stagingURL(for filename: String) -> URL {
        let safe = filename.replacingOccurrences(of: "/", with: "_")
        return directory.appendingPathComponent("\(UUID().uuidString)_\(safe)")
    }

    /// Temp dir for per-part chunk files backing background upload tasks.
    /// Also in Application Support (not the OS-purged tmp dir) so a chunk
    /// survives until its background task completes, even across suspension.
    static var partsDirectory: URL {
        let base = try! FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = base.appendingPathComponent("QuickUpload/parts", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        var d = dir
        var rv = URLResourceValues(); rv.isExcludedFromBackup = true
        try? d.setResourceValues(rv)
        return dir
    }
}

// MARK: - Background upload session

/// Identifies which (upload, part) a background URLSession task belongs to.
/// Stored in the task's `taskDescription` so it survives app suspension and
/// relaunch — that's how a delegate callback maps back to an upload even when
/// the task was started in a previous process.
struct UploadTaskTag: Codable {
    let stateId: String
    let partNumber: Int
    let size: Int64

    func encoded() -> String {
        guard let data = try? JSONEncoder().encode(self),
              let s = String(data: data, encoding: .utf8) else { return "" }
        return s
    }

    static func decode(_ s: String?) -> UploadTaskTag? {
        guard let s, let data = s.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(UploadTaskTag.self, from: data)
    }
}

/// Delegate for the background upload URLSession. Background sessions can't use
/// the async/await convenience methods — part tasks finish via these callbacks,
/// possibly after the app was suspended and relaunched. Runs on the session's
/// private delegate queue; every callback hops to the main actor to mutate
/// UploadManager (the single source of truth).
final class BackgroundUploadDelegate: NSObject, URLSessionDataDelegate {

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        guard let tag = UploadTaskTag.decode(task.taskDescription) else { return }
        // HTTP error status arrives as a successful task with a non-2xx
        // response, NOT as `error` (which is transport-level). Check both.
        let statusCode = (task.response as? HTTPURLResponse)?.statusCode ?? 0
        let errored = error != nil
        Task { @MainActor in
            UploadManager.shared.handlePartTaskCompletion(
                tag: tag, statusCode: statusCode, errored: errored)
        }
    }

    /// Fires once the session has delivered all queued completion callbacks
    /// after a background relaunch. We then call the system-provided
    /// completion handler (stashed by the AppDelegate) so iOS can snapshot the
    /// UI and re-suspend us.
    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor in
            UploadManager.shared.finishBackgroundEvents()
        }
    }
}
