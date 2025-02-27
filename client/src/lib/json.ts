export class StrictJSON {
    static validator(jsonString: string): boolean {
        // Parse the JSON string
        try {
            const parsed = JSON.parse(jsonString)

            // Ensure it's an object with outer {} brackets.
            if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
                // Must be an object with outer {} brackets
                return false
            }

            // Ensure it has at least one key.
            if (Object.keys(parsed).length === 0) {
                // Object must have at least one key
                return false
            }
        } catch(err) {
            // Return false if any other errors arise from the
            // JSON.parse method
            return false
        }
        // Valid JSON
        return true
    }
}
