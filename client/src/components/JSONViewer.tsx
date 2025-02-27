import { cn } from "../lib/utils"

interface PreViewerProps {
    value: string
    className?: string
}

export default function PreViewer({ value, className }: PreViewerProps) {
    return (
        <div className={cn(
            "p-2",
            "font-mono text-sm text-white",
            "bg-neutral-900",
            "border-3 rounded-lg border-gray-300",
            "max-h-screen overflow-auto",
            className)}
        >
            <pre className={className}>{value}</pre>
        </div>
    )
}
