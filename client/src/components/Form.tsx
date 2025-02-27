import { cn } from "../lib/utils"

interface CustomFormProps {
    children: React.ReactNode
    onSubmit?: (e: React.FormEvent<HTMLFormElement>) => void
    className?: string
}

export default function Form({ children, onSubmit, className }: CustomFormProps) {
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
        >
            {children}
        </form>
    )
}
