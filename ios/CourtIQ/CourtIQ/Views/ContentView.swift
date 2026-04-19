import SwiftUI

struct ContentView: View {
    @StateObject private var session = TennisSession()

    var body: some View {
        ZStack {
            CameraPreviewView(session: session)
                .ignoresSafeArea()

            PoseOverlayView(session: session)
                .ignoresSafeArea()

            VStack {
                SessionHeaderView(session: session)
                Spacer()
                RecordingControlsView(session: session)
            }
            .padding()
        }
        .onAppear {
            session.startCamera()
        }
    }
}

struct SessionHeaderView: View {
    @ObservedObject var session: TennisSession

    var body: some View {
        HStack(spacing: 16) {
            if session.isRecording {
                HStack(spacing: 6) {
                    Circle()
                        .fill(.red)
                        .frame(width: 10, height: 10)
                    Text(session.recordingDuration)
                        .font(.system(.body, design: .monospaced))
                        .foregroundColor(.white)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.black.opacity(0.6))
                .cornerRadius(8)
            }

            Spacer()

            ShotCounterView(session: session)
        }
        .padding(.top, 8)
    }
}

struct ShotCounterView: View {
    @ObservedObject var session: TennisSession

    var body: some View {
        HStack(spacing: 12) {
            shotPill("S", count: session.serveCount, color: .orange)
            shotPill("FH", count: session.forehandCount, color: .green)
            shotPill("BH", count: session.backhandCount, color: .blue)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.black.opacity(0.6))
        .cornerRadius(8)
    }

    func shotPill(_ label: String, count: Int, color: Color) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(color)
            Text("\(count)")
                .font(.system(size: 14, weight: .bold, design: .monospaced))
                .foregroundColor(.white)
        }
    }
}

struct RecordingControlsView: View {
    @ObservedObject var session: TennisSession

    var body: some View {
        HStack(spacing: 40) {
            Button(action: { session.toggleRecording() }) {
                ZStack {
                    Circle()
                        .stroke(.white, lineWidth: 4)
                        .frame(width: 72, height: 72)
                    if session.isRecording {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(.red)
                            .frame(width: 28, height: 28)
                    } else {
                        Circle()
                            .fill(.red)
                            .frame(width: 60, height: 60)
                    }
                }
            }
        }
        .padding(.bottom, 20)
    }
}
