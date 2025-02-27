import { cn } from "../lib/utils"

interface FlashProps {
    message: string
    className?: string
}

export default function Flash({ message, className }: FlashProps) {
    return (
        <div
            className={cn(
                "bg-neutral-800",
                "border border-neutral-500 rounded-md",
                "p-2",
                "text-white",
                "overflow-auto",
                className
            )}
        >
            {message}
        </div>
    )
}
