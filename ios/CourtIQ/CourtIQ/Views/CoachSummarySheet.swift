import SwiftUI

/// Native renderer for the coaching JSON that the WebView gallery would
/// otherwise show inside an HTML modal. WebView's "COACH" button posts
/// the cached payload over the openCoach JS bridge; we present this in
/// a SwiftUI sheet on top of the WebView.
///
/// The payload shape mirrors the gallery's `coachCache` content:
/// {
///   headline: String,
///   working: [Section]?,
///   improve: [Section]?,
///   focus:   [Section]?
/// }
/// Section = { point, detail, examples?: [{ t, type, note? }] }
struct CoachSummarySheet: View {
    let videoId: String
    let payload: CoachPayload
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    Text(videoId)
                        .font(.caption.monospaced())
                        .foregroundColor(.brandTextSecondary)

                    if let head = payload.headline, !head.isEmpty {
                        Text(head)
                            .font(.title3.weight(.semibold))
                            .foregroundColor(.brandText)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    sectionView(title: "What's working", items: payload.working)
                    sectionView(title: "What to improve", items: payload.improve)
                    sectionView(title: "Focus on", items: payload.focus)
                }
                .padding(.horizontal, 20)
                .padding(.top, 8)
                .padding(.bottom, 32)
            }
            .background(Color.brandBackground)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Text("Coach summary")
                        .font(.headline)
                        .foregroundColor(.brandText)
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                        .foregroundColor(.brandAccent)
                }
            }
        }
    }

    @ViewBuilder
    private func sectionView(title: String, items: [CoachItem]?) -> some View {
        if let items, !items.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Text(title.uppercased())
                    .font(.caption.weight(.heavy))
                    .tracking(0.8)
                    .foregroundColor(.brandAccent)
                ForEach(items) { item in itemRow(item) }
            }
        }
    }

    private func itemRow(_ item: CoachItem) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(item.point ?? "")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.brandText)
            if let detail = item.detail, !detail.isEmpty {
                Text(detail)
                    .font(.subheadline)
                    .foregroundColor(.brandTextSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let examples = item.examples, !examples.isEmpty {
                FlowLayout(spacing: 6) {
                    ForEach(examples) { ex in
                        Text(formattedExample(ex))
                            .font(.caption.weight(.medium))
                            .foregroundColor(.brandBackground)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .background(Color.brandAccent.opacity(0.85))
                            .clipShape(Capsule())
                    }
                }
                .padding(.top, 2)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.brandSurface)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func formattedExample(_ ex: CoachExample) -> String {
        let ts = ex.t.flatMap { secondsToMSS($0) } ?? "—"
        let type = ex.type ?? ""
        if let n = ex.note, !n.isEmpty { return "\(ts) \(type) — \(n)" }
        return "\(ts) \(type)"
    }

    private func secondsToMSS(_ t: Double) -> String {
        let m = Int(t) / 60
        let s = Int(t) % 60
        return String(format: "%d:%02d", m, s)
    }
}

// Wire-types. Codable so we can decode the WebView JS payload directly.
struct CoachPayload: Codable, Equatable {
    let headline: String?
    let working: [CoachItem]?
    let improve: [CoachItem]?
    let focus: [CoachItem]?
}

struct CoachItem: Codable, Equatable, Identifiable {
    let point: String?
    let detail: String?
    let examples: [CoachExample]?
    var id: String { (point ?? "") + (detail ?? "") }
}

struct CoachExample: Codable, Equatable, Identifiable {
    let t: Double?
    let type: String?
    let note: String?
    var id: String { "\(t ?? -1)-\(type ?? "")" }
}

/// Simple flow layout that wraps capsule chips onto multiple rows.
/// SwiftUI's built-in `Layout` protocol is the cleanest option here.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var rowWidth: CGFloat = 0
        var rowHeight: CGFloat = 0
        var totalHeight: CGFloat = 0
        var maxRowWidth: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if rowWidth + s.width + (rowWidth > 0 ? spacing : 0) > width {
                totalHeight += rowHeight + spacing
                maxRowWidth = max(maxRowWidth, rowWidth)
                rowWidth = s.width
                rowHeight = s.height
            } else {
                rowWidth += s.width + (rowWidth > 0 ? spacing : 0)
                rowHeight = max(rowHeight, s.height)
            }
        }
        totalHeight += rowHeight
        maxRowWidth = max(maxRowWidth, rowWidth)
        return CGSize(width: maxRowWidth, height: totalHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let width = bounds.width
        var x: CGFloat = bounds.minX
        var y: CGFloat = bounds.minY
        var rowHeight: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x + s.width - bounds.minX > width {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            v.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(s))
            x += s.width + spacing
            rowHeight = max(rowHeight, s.height)
        }
    }
}
