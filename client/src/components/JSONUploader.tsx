import Input from "./Input"
import { StrictJSON } from "../lib/json"
import { cn } from "../lib/utils"

interface JSONUploaderProps {
    onChange: (value: string) => void
    onError: (error: string) => void
    className?: string
}

export default function JSONUploader({ onChange, onError, className }: JSONUploaderProps) {
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            // Only allow one file as upload!
            const file = e.target.files[0]
            const reader = new FileReader()
            reader.onload = () => {
                const fileContent = reader.result
                if (typeof fileContent === "string") {
                    // Validate that it is JSON before changing value
                    if (!StrictJSON.validator(fileContent)) {
                        onError("Invalid JSON, must be enclosed by {} and have at least on key")
                        onChange("")
                        return
                    }
                    onError("")
                    onChange(fileContent)
                }
            }
            reader.onerror = () => {
                const err = reader.error
                    ? reader.error.message
                    : "FileReader: unknown error occurred"
                onError(err)
                onChange("")
            }
            reader.readAsText(file)
        }
    }

    return (
        <Input
            type="file"
            accept="application/json"
            onChange={handleFileChange}
            className={cn(className)}
        />
    )
}
