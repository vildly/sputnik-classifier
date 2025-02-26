import { useEffect, useState } from "react"
import { cn } from "../lib/utils"

interface JSONViewerProps {
    data: string
    errorCallback?: (error: string | null) => void
    className?: string
}

export default function JSONViewer({ data, errorCallback, className }: JSONViewerProps) {
    const [error, setError] = useState<string | null>(null)
    const [formatted, setFormatted] = useState<string>("")

    useEffect(() => errorCallback && errorCallback(error), [error])
    useEffect(() => handleDataChange(), [data])

    function handleDataChange() {
        try {
            const parsed = JSON.parse(data)
            setFormatted(JSON.stringify(parsed, null, 2))
            setError(null)
        } catch (err) {
            setError("Incorrect JSON format")
        }
    }

    return (
        <div className={cn(
            "p-2",
            "font-mono text-sm text-white",
            "bg-neutral-900",
            "border-3 rounded-lg border-gray-300",
            className)}
        >
            <pre className={cn(className)}>{formatted}</pre>
        </div>
    )
}
