export class StrictJSON {
    /**
     * Custom JSON parser to ensure the input is valid JSON with outer `{}` brackets
     * and at least one key.
     * @param jsonString - The JSON string to parse.
     * @param errorCallback - Optional callback to receive the error message or null.
     * @returns True if the JSON is valid; false otherwise.
     */
    static validator(jsonString: string, errorCallback?: (error: string | null) => void): boolean {
        try {
            // Parse the JSON string
            const parsed = JSON.parse(jsonString)

            // Ensure it's an object with outer {} brackets.
            if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
                const err = "Must be an object with outer {} brackets"
                errorCallback && errorCallback(err)
                return false
            }

            // Ensure it has at least one key.
            if (Object.keys(parsed).length === 0) {
                const err = "Object must have at least one key"
                errorCallback && errorCallback(err)
                return false
            }

            // Valid JSON; notify via callback with no error.
            errorCallback && errorCallback(null)
            return true
        } catch (err: any) {
            errorCallback && errorCallback(err.message)
            return false
        }
    }
}
