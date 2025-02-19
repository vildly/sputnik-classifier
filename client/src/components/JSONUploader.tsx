import { useEffect, useState } from "react"
import { cn } from "../lib/utils"

interface JSONUploaderProps {
    errorCB?: (error: string | null) => void
    setValue: (value: string) => void
    label: string
}

export default function JSONUploader({ errorCB, setValue, label }: JSONUploaderProps) {
    const [error, setError] = useState<string | null>(null)

    useEffect(() => errorCB && errorCB(error), [error])

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            // Only allow one file as upload!
            const file = e.target.files[0]
            const reader = new FileReader()

            reader.onload = () => {
                const fileContent = reader.result
                if (typeof fileContent === "string") {
                    try {
                        const parsed = JSON.parse(fileContent)
                        const formatted = JSON.stringify(parsed, null, 2)
                        setValue(formatted)
                        setError(null)
                    } catch (err) {
                        setError("Invalid JSON file")
                    }
                }
            }

            reader.onerror = () => {
                setError("Error reading file")
            }

            reader.readAsText(file)
        }
    }

    return (
        <div className="flex flex-col space-y-2">
            <label className="text-white text-lg">
                {error ? <span className="text-red-500">{error}</span> : label}
            </label>
            <input
                type="file"
                accept="application/json"
                onChange={handleFileChange}
                className={cn("bg-neutral-900 p-2 rounded-lg border-gray-300 text-white")}
            />
        </div>
    )
}
