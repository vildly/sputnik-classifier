import { useState } from "react"
import JSONUploader from "../components/JSONUploader"
import JSONEditor from "../components/JSONEditor"
import Input from "../components/Input"
import Loading from "../components/Loading"
import { HexColors } from "../lib/colors"
import { sendRaw } from "../services/api"
import Error from "../components/Error"
import Form from "../components/Form"

export default function UploadView() {
    const [error, setError] = useState<string | null>(null)
    const [processing, setProcessing] = useState<boolean>(false)
    const [value, setValue] = useState<string>("")

    function handleError(error: string | null): void {
        setError(error)
    }

    function handleSubmit(): void {
        // Prevent incorrect JSON
        if (error) return
        // Prevent re-entry
        if (processing) return

        setProcessing(true)
        sendRaw(value)

        // Reset
        setValue("")
        setProcessing(false)
    }

    return (
        <div className="flex flex-col space-y-2">
            {error && <Error message={error} />}
            <Form onSubmit={handleSubmit}>
                <JSONUploader
                    errorCallback={handleError}
                    valueCallback={setValue}
                />
                {value && <JSONEditor
                    errorCallback={handleError}
                    valueCallback={setValue}
                    value={value}
                />}
                {(!error && value) && <Input
                    type="submit"
                    value="Submit"
                    className="cursor-pointer my-2"
                />}
                {processing && <Loading color={HexColors.WHITE} />}
            </Form>
        </div>
    )
}
