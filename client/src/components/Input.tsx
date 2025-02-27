import { cn } from "../lib/utils"

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    className?: string
}

export default function Input({ className, ...props }: InputProps) {
    return (
        <input
            // The className prop must be at the end for any custom changes to apply correctly!
            className={cn("w-fit px-4 py-2 text-white bg-blue-500 rounded hover:bg-blue-600 transition", className)}
            {...props}
        />
    )
}
