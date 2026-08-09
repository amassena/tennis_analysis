import Foundation
import Security

/// Keychain-backed storage for our server-issued JWT.
///
/// Stored under service `com.playfullife.courtiq.token` with
/// `kSecAttrAccessibleAfterFirstUnlock` so background URLSession tasks
/// can read the token after the device unlocks once post-reboot.
enum TokenStore {
    private static let service = "com.playfullife.courtiq.token"
    private static let account = "default"

    static func save(_ jwt: String) throws {
        let data = Data(jwt.utf8)

        // Delete any existing entry first; SecItemUpdate is fussy and overkill here.
        let deleteQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(deleteQuery as CFDictionary)

        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock,
            kSecValueData as String: data,
        ]
        let status = SecItemAdd(addQuery as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw TokenStoreError.keychain(status: status)
        }
    }

    static func load() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess,
              let data = result as? Data,
              let jwt = String(data: data, encoding: .utf8)
        else {
            return nil
        }
        return jwt
    }

    static func clear() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}

enum TokenStoreError: Error, LocalizedError {
    case keychain(status: OSStatus)

    var errorDescription: String? {
        switch self {
        case .keychain(let status):
            return "Keychain error: \(status)"
        }
    }
}
