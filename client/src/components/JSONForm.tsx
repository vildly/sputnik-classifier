import { useState } from "react"
import JSONUploader from "./JSONUploader"
import JSONEditor from "./JSONEditor"
import Input from "./Input"
import Loading from "./Loading"
import { HexColors } from "../lib/colors"

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
                <JSONUploader errorCallback={handleError} valueCallback={setValue} />
                {value && <JSONEditor errorCallback={handleError} valueCallback={setValue} value={value} />}
                {(!error && value) && <Input type="submit" value="Submit" className="cursor-pointer" />}
                {processing && <Loading color={HexColors.WHITE} />}
            </form>
        </div>
    )
}
