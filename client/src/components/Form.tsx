import { cn } from "../lib/utils"

interface FormProps extends React.FormHTMLAttributes<HTMLFormElement> {
    children: React.ReactNode
    onSubmit?: (e: React.FormEvent<HTMLFormElement>) => void
    className?: string
}

export default function Form({ children, onSubmit, className, ...props }: FormProps) {
    // This base handleSubmit always prevents the default form submission behavior.
    const handleSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
        e.preventDefault()
        if (onSubmit) {
            onSubmit(e)
        }
    }

    return (
        <form
            onSubmit={handleSubmit}
            className={cn("flex flex-col", className)}
            {...props}
        >
            {children}
        </form>
    )
}
