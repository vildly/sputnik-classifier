import { useState } from "react"
import { cn } from "../lib/utils"

interface InputJSONProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
    label?: string
}

export default function InputJSON({ label = "Label", placeholder = "...", ...props }: InputJSONProps) {
    const [value, setValue] = useState<string>("")
    const [error, setError] = useState<string | null>(null)

    // Update state as the user types into the textarea
    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setValue(e.target.value)
    }

    // When the textarea loses focus, try to parse and reformat the JSON
    const handleBlur = (_: React.FocusEvent<HTMLTextAreaElement>) => {
        try {
            const parsed = JSON.parse(value)
            const formatted = JSON.stringify(parsed, null, 2)
            setValue(formatted)
            setError(null)
        } catch (err) {
            setError("Invalid JSON")
        }
    }

    return (
        <div className="flex flex-col items-center p-6 w-full h-screen">
            <label className="my-1 text-white text-lg">
                {error
                ? <span className="my-1 text-red-500 text-lg">{error}</span>
                : label}
            </label>
            <textarea
                {...props}
                value={value}
                placeholder={placeholder}
                onChange={handleChange}
                onBlur={handleBlur}
                className={cn(
                    "p-2",
                    "resize-none [width:80ch] h-full",
                    "text-white text-md",
                    "bg-neutral-900",
                    error ? "border-red-500" : "border-gray-300",
                    "border-3 rounded-lg",
                    "focus:outline-none focus:ring-2 focus:ring-blue-500",
                    "transition"
                )}
            />
        </div>
    )
}
