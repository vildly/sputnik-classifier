import { cn } from "../lib/utils"

interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement> {
    label: string
    className?: string
}

export default function Label({ label, className, ...props }: LabelProps) {
    return (
        <label {...props} className={cn("text-white text-lg", className)}>{label}</label>
    )
}
