import { cn } from "../lib/utils"

export interface SpacerProps {
    className?: string
}

export default function Spacer({ className }: SpacerProps) {
    return (
        <div className="w-full">
            <div className={cn("mx-auto w-11/12 border-t-2 rounded-full border-neutral-900", className)} />
        </div>
    )
}
