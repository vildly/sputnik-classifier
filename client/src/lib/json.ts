export class StrictJSON {
    /**
     * Custom JSON parser to ensure the input is valid JSON with outer `{}` brackets
     * and at least one key.
     * @param jsonString - The JSON string to parse.
     * @returns The parsed object.
     * @throws Error if the input is not valid or doesn't meet the criteria.
     */
    static parse(jsonString: string): Record<string, any> {
        // Parse the JSON string
        const parsed = JSON.parse(jsonString)

        // Ensure it's an object with outer {} brackets
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
            throw new Error("StrictJSON.parse: must be an object with outer {} brackets")
        }

        // Ensure it has at least one key
        if (Object.keys(parsed).length === 0) {
            throw new Error("StrictJSON.parse: object must have at least one key")
        }

        return parsed
    }
}
