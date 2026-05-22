import SwiftUI

struct UploadRowView: View {
    let state: UploadState
    var onRetry: () -> Void
    var onDiscard: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: iconName)
                    .foregroundColor(iconColor)
                    .font(.system(size: 18, weight: .semibold))

                VStack(alignment: .leading, spacing: 2) {
                    Text(state.filename)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                    Text(secondaryText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                switch state.status {
                case .failed:
                    HStack(spacing: 8) {
                        Button("Retry", action: onRetry)
                            .font(.caption.weight(.semibold))
                        Button(action: onDiscard) {
                            Image(systemName: "trash")
                        }
                        .foregroundColor(.red)
                    }
                case .completed:
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                default:
                    Text(percentText)
                        .font(.caption.monospacedDigit())
                        .foregroundColor(.secondary)
                }
            }

            if state.status != .completed && state.status != .failed {
                ProgressView(value: state.progress)
                    .progressViewStyle(.linear)
            }

            if let err = state.errorMessage, state.status == .failed {
                Text(err)
                    .font(.caption2)
                    .foregroundColor(.red)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }

    private var iconName: String {
        switch state.status {
        case .queued, .initializing: return "clock"
        case .uploading: return "arrow.up.circle"
        case .finalizing: return "hourglass"
        case .completed: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    private var iconColor: Color {
        switch state.status {
        case .failed: return .red
        case .completed: return .green
        default: return .accentColor
        }
    }

    private var secondaryText: String {
        switch state.status {
        case .queued: return "Queued"
        case .initializing: return "Starting…"
        case .uploading:
            let mbDone = Double(state.bytesUploaded) / 1_048_576
            let mbTotal = Double(state.totalBytes) / 1_048_576
            return String(format: "%.0f / %.0f MB", mbDone, mbTotal)
        case .finalizing: return "Finishing…"
        case .completed: return "Uploaded"
        case .failed: return "Failed"
        }
    }

    private var percentText: String {
        "\(Int(state.progress * 100))%"
    }
}
