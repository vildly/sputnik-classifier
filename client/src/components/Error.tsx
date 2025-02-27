import { cn } from "../lib/utils"

interface ErrorProps {
    message: string
    className?: string
}

export default function Error({ message, className }: ErrorProps) {
    return (
        <div
            className={cn(
                "bg-red-800",
                "border border-red-500 rounded-md",
                "p-2",
                "text-white",
                "overflow-auto",
                className
            )}
        >{message}</div>
    )
}
