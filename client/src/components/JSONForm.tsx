import { useState } from "react"
import JSONUploader from "./JSONUploader"
import JSONInput from "./JSONInput"

export default function JSONForm() {
    const [error, setError] = useState<string | null>(null)
    const [processing, setProcessing] = useState<boolean>(false)
    const [value, setValue] = useState<string>("")

    function handleError(error: string | null): void {
        setError(error)
    }

    async function handleSubmit(e: React.FormEvent<HTMLFormElement>): Promise<void> {
        // Prevent default behaviour
        e.preventDefault()
        // Prevent incorrect JSON
        if (error) return
        // Prevent re-entry
        if (processing) return

        setProcessing(true)

        try {
            console.log(`Submitted value: ${value}`);
            // TODO: Use service here instead later!
        } finally {
            // Reset
            setValue("")
            setProcessing(false)
        }
    }

    return (
        <div className="w-full h-full">
            <form onSubmit={handleSubmit} className="h-full flex flex-col space-y-2">
                <JSONUploader
                    errorCB={handleError}
                    setValue={setValue}
                    label="Upload JSON file"
                />
                <JSONInput
                    errorCB={handleError}
                    setValue={setValue}
                    value={value}
                    label="Edit JSON content"
                />
                <button
                    type="submit"
                    className="w-fit px-4 py-2 text-white bg-blue-500 rounded hover:bg-blue-600 transition"
                >
                    Submit
                </button>
            </form>
        </div>
    )
}
