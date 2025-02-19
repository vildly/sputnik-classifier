import { useEffect, useState } from "react"
import { cn } from "../lib/utils"

interface JSONInputProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
    errorCB?: (error: string | null) => void
    setValue: (value: string) => void
    value: string
    label: string
}

export default function JSONInput({ errorCB, setValue, value, label, ...props }: JSONInputProps) {
    const [error, setError] = useState<string | null>(null)

    useEffect(() => errorCB && errorCB(error), [error])

    // Update state as the user types into the textarea
    function handleChange(e: React.ChangeEvent<HTMLTextAreaElement>): void {
        setValue(e.target.value)
    }

    // When the textarea loses focus, try to parse and reformat the JSON
    function handleBlur(_: React.FocusEvent<HTMLTextAreaElement>): void {
        try {
            const parsed = JSON.parse(value)
            const formatted = JSON.stringify(parsed, null, 2)
            setValue(formatted)
            setError(null)
        } catch (err) {
            setError("Incorrect JSON format")
        }
    }

    return (
        <div className="h-full flex flex-col space-y-2">
            <label className="text-white text-lg">{label}</label>
            <textarea
                {...props}
                value={value}
                onChange={handleChange}
                onBlur={handleBlur}
                className={cn(
                    "h-full",
                    "p-2 resize-none",
                    "text-md text-white",
                    "bg-neutral-900",
                    "border-3 rounded-lg",
                    error ? "border-red-500" : "border-gray-300",
                    "focus:outline-none focus:ring-2 focus:ring-blue-500",
                    "transition duration-300"
                )}
            />
        </div>
    )
}
