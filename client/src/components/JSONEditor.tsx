import { useEffect, useState } from "react"
import Label from "./Label"
import Textarea from "./Textarea"
import { cn } from "../lib/utils"

interface JSONEditorProps {
    errorCallback?: (error: string | null) => void
    valueCallback: (value: string) => void
    value: string
}

export default function JSONEditor({ errorCallback, valueCallback, value }: JSONEditorProps) {
    const [error, setError] = useState<string | null>(null)

    useEffect(() => errorCallback && errorCallback(error), [error])
    useEffect(() => parseValue(), [value])

    function parseValue(): void {
        try {
            JSON.parse(value)
            setError(null)
        } catch (err) {
            setError("Incorrect JSON format")
        }
    }

    // Update state as the user types into the textarea
    function handleChange(e: React.ChangeEvent<HTMLTextAreaElement>): void {
        valueCallback(e.target.value)
    }

    // When the textarea loses focus, try to parse and reformat the JSON
    function handleBlur(_: React.FocusEvent<HTMLTextAreaElement>): void {
        try {
            const parsed = JSON.parse(value)
            valueCallback(JSON.stringify(parsed, null, 2))
            setError(null)
        } catch (err) {
            setError("Incorrect JSON format")
        }
    }

    return (
        <div className="flex flex-col">
            <Label label="Edit JSON" />
            <Textarea
                value={value}
                onBlur={handleBlur}
                onChange={handleChange}
                className={cn(error && "border-red-500")}
            />
        </div>
    )
}
