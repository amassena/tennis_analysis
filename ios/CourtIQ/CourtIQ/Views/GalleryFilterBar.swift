import SwiftUI

/// UX-4 — native filter bar that floats above the gallery WebView and
/// drives it via `applyNativeFilter()` JS calls. Replaces the WebView's
/// own desktop-style dropdown row (which is auto-hidden on iOS).
///
/// State design: filter + sort identifiers match exactly what the
/// gallery's JS expects (`currentFilter` / `currentSort` globals), so
/// the bridge is a single `webView.evaluateJavaScript("applyNativeFilter({...})")`.
struct GalleryFilterBar: View {
    @Binding var state: FilterState
    var onPick: (FilterState) -> Void

    @State private var showingSheet = false

    var body: some View {
        Button { showingSheet = true } label: {
            HStack(spacing: 10) {
                pill(label: state.filter.displayName, systemImage: "line.3.horizontal.decrease.circle")
                pill(label: state.sort.displayName, systemImage: "arrow.up.arrow.down.circle")
                Spacer()
                Image(systemName: "chevron.down")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.brandTextSecondary)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color.brandSurface)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 6)
        .background(Color.brandBackground)
        .sheet(isPresented: $showingSheet) {
            GalleryFilterSheet(state: $state) { newState in
                onPick(newState)
            }
            .presentationDetents([.medium])
        }
    }

    private func pill(label: String, systemImage: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
                .font(.subheadline)
                .foregroundColor(.brandAccent)
            Text(label)
                .font(.subheadline.weight(.medium))
                .foregroundColor(.brandText)
                .lineLimit(1)
        }
    }
}

struct GalleryFilterSheet: View {
    @Binding var state: FilterState
    var onApply: (FilterState) -> Void
    @Environment(\.dismiss) private var dismiss

    @State private var draft: FilterState = FilterState()

    var body: some View {
        NavigationView {
            Form {
                Section {
                    Picker("Type", selection: $draft.filter) {
                        ForEach(GalleryFilterType.allCases) { f in
                            Text(f.displayName).tag(f)
                        }
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("Show").foregroundColor(.brandTextSecondary)
                }

                Section {
                    Picker("Sort by", selection: $draft.sort) {
                        ForEach(GallerySortOption.allCases) { s in
                            Text(s.displayName).tag(s)
                        }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                } header: {
                    Text("Sort").foregroundColor(.brandTextSecondary)
                }

                Section {
                    Button(role: .destructive) {
                        draft = FilterState()
                    } label: {
                        Label("Reset to defaults", systemImage: "arrow.counterclockwise")
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .background(Color.brandBackground)
            .navigationTitle("Filter & Sort")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") { dismiss() }
                        .foregroundColor(.brandTextSecondary)
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Apply") {
                        state = draft
                        onApply(draft)
                        dismiss()
                    }
                    .foregroundColor(.brandAccent)
                    .fontWeight(.semibold)
                }
            }
        }
        .onAppear { draft = state }
    }
}

// MARK: - State model

struct FilterState: Equatable {
    var filter: GalleryFilterType = .all
    var sort: GallerySortOption = .recordedDesc

    /// JS literal posted into the WebView as `applyNativeFilter(<this>)`.
    func toJSObject() -> String {
        "{filter:\"\(filter.rawValue)\",sort:\"\(sort.rawValue)\"}"
    }
}

enum GalleryFilterType: String, CaseIterable, Identifiable {
    case all       = "all"
    case serves    = "serve"
    case forehands = "forehand"
    case backhands = "backhand"

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .all:       return "All"
        case .serves:    return "Serves"
        case .forehands: return "Forehands"
        case .backhands: return "Backhands"
        }
    }
}

enum GallerySortOption: String, CaseIterable, Identifiable {
    case recordedDesc = "recorded-desc"
    case recordedAsc  = "recorded-asc"
    case shotsDesc    = "shots-desc"
    case shotsAsc     = "shots-asc"
    case durationDesc = "duration-desc"
    case durationAsc  = "duration-asc"

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .recordedDesc: return "Newest first"
        case .recordedAsc:  return "Oldest first"
        case .shotsDesc:    return "Most shots"
        case .shotsAsc:     return "Fewest shots"
        case .durationDesc: return "Longest"
        case .durationAsc:  return "Shortest"
        }
    }
}
