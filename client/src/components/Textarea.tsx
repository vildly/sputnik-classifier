import { cn } from "../lib/utils"

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
    className?: string
}

export default function Textarea({ className, ...props }: TextareaProps) {
    return (
        <textarea
            className={cn("p-2 font-mono text-sm text-white bg-neutral-900 border-3 rounded-lg border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300", className)}
            {...props}
        />
    )
}
