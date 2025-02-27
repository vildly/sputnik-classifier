import { useEffect, useState } from "react"
import Input from "./Input"
import { StrictJSON } from "../lib/json"
import { cn } from "../lib/utils"

interface JSONUploaderProps {
    errorCallback?: (error: string | null) => void
    onChange: (value: string) => void
}

export default function JSONUploader({ errorCallback, onChange }: JSONUploaderProps) {
    const [error, setError] = useState<string | null>(null)

    useEffect(() => errorCallback && errorCallback(error), [error])

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            // Only allow one file as upload!
            const file = e.target.files[0]
            const reader = new FileReader()

            reader.onload = () => {
                const fileContent = reader.result
                if (typeof fileContent === "string") {
                    try {
                        // Validate that it is JSON before changing value
                        if (!StrictJSON.validator(fileContent, setError)) return
                        onChange(JSON.stringify(JSON.parse(fileContent), null, 2))
                        setError(null)
                    } catch(err: any) {
                        setError(err.message)
                    }
                }
            }

            reader.onerror = () => {
                const err = reader.error
                    ? reader.error.message
                    : "FileReader: unknown error occurred"
                setError(err)
            }

            reader.readAsText(file)
        }
    }

    return (
        <Input
            type="file"
            accept="application/json"
            onChange={handleFileChange}
            className={cn(
                error && "bg-red-500",
                "cursor-pointer"
            )}
        />
    )
}
