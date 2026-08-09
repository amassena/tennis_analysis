import Foundation

/// Polls `/api/status/<vid>` to surface server-side pipeline progress
/// after the iOS-side chunked upload completes.
///
/// Backoff: 5 / 10 / 20 / 40 / 80 / 120s (then steady at 120s) until a
/// terminal status is reached. Cancels if the caller goes away or the
/// app is killed; UploadResumer re-arms pollers for not-yet-terminal
/// uploads on next launch.
@MainActor
final class StatusPoller {
    static let shared = StatusPoller()

    /// Video IDs we're actively polling.
    private var active: Set<String> = []

    enum ServerStatus: String, Codable {
        case awaiting_coordinator
        case queued
        case running
        case completed
        case failed
    }

    struct StatusResponse: Decodable {
        let id: String?
        let status: String?
        let stage: String?
        let progress: Double?
        let video_url: String?
        let error: String?
    }

    func start(videoId: String, onUpdate: @escaping (StatusResponse) -> Void) {
        guard !active.contains(videoId) else { return }
        active.insert(videoId)
        Task { [weak self] in
            await self?.poll(videoId: videoId, onUpdate: onUpdate)
        }
    }

    func cancel(videoId: String) {
        active.remove(videoId)
    }

    private func poll(videoId: String, onUpdate: @escaping (StatusResponse) -> Void) async {
        let delays: [UInt64] = [5, 10, 20, 40, 80, 120].map { UInt64($0) * 1_000_000_000 }
        var attempt = 0
        while active.contains(videoId) {
            try? await Task.sleep(nanoseconds: delays[min(attempt, delays.count - 1)])
            guard active.contains(videoId) else { return }
            do {
                let resp: StatusResponse = try await APIClient.get(
                    path: "api/status/\(videoId)",
                    requireAuth: false  // status is public for now
                )
                onUpdate(resp)
                if let s = resp.status, s == "completed" || s == "failed" {
                    active.remove(videoId)
                    return
                }
            } catch {
                // Network blips are fine — keep polling.
            }
            attempt += 1
        }
    }
}
