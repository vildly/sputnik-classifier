import { useEffect, useState } from "react"

interface JSONViewerProps {
    data: string
}

export default function JSONViewer({ data }: JSONViewerProps) {
    const [formatted, setFormatted] = useState<string>("")

    useEffect(() => {
        setFormatted(JSON.stringify(data, null, 2))
    }, [data])

    return (
        <div>
            <pre>{formatted}</pre>
        </div>
    )
}
