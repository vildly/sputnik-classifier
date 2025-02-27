import { useEffect, useState } from "react"
import Input from "./Input"
import { StrictJSON } from "../lib/json"

interface JSONUploaderProps {
    errorCallback?: (error: string | null) => void
    valueCallback: (value: string) => void
}

export default function JSONUploader({ errorCallback, valueCallback }: JSONUploaderProps) {
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
                        const parsed = StrictJSON.parse(fileContent)
                        const formatted = JSON.stringify(parsed, null, 2)
                        valueCallback(formatted)
                        setError(null)
                    } catch(err: any) {
                        valueCallback(fileContent)
                        setError(err.message)
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
        <Input
            type="file"
            accept="application/json"
            onChange={handleFileChange}
            className="cursor-pointer"
        />
    )
}
