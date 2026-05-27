import Foundation
import Combine
import SwiftUI

/// One row in `GET /api/u/<hash>/recent`.
struct RecentUpload: Codable, Identifiable, Equatable {
    let video_id: String
    let filename: String?
    let status: String
    let stage: String?
    let progress: Int?
    let uploaded_at: String?
    let updated_at: String?
    let error: String?
    let gallery_url: String?

    var id: String { video_id }

    var isComplete: Bool { status == "complete" }
    var isFailed: Bool { status == "failed" }
    var isInflight: Bool { !isComplete && !isFailed }
}

private struct RecentResponse: Codable {
    let user_hash: String
    let count: Int
    let items: [RecentUpload]
}

/// Drives the "Recently uploaded" section in the Upload tab. Polls the
/// worker every `pollInterval` seconds while the view is foregrounded.
@MainActor
final class RecentUploadsModel: ObservableObject {
    @Published private(set) var items: [RecentUpload] = []
    @Published private(set) var lastError: String?
    @Published private(set) var isLoading = false

    let userHash: String
    private var timer: AnyCancellable?
    private let pollInterval: TimeInterval = 15

    init(userHash: String) {
        self.userHash = userHash
    }

    func start() {
        Task { await self.refresh() }
        timer = Timer.publish(every: pollInterval, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                Task { await self?.refresh() }
            }
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }

    func refresh() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let resp: RecentResponse = try await APIClient.get(
                path: "api/u/\(userHash)/recent?limit=25",
            )
            // Avoid spurious view churn when nothing actually changed.
            if resp.items != items { items = resp.items }
            lastError = nil
        } catch APIClient.APIError.unauthorized {
            lastError = "Session expired"
        } catch {
            lastError = error.localizedDescription
        }
    }
}
