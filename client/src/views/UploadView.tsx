import { useState } from "react"
import JSONUploader from "../components/JSONUploader"
import Input from "../components/Input"
import Loading from "../components/Loading"
import { HexColors } from "../lib/colors"
import { sendRaw } from "../services/api"
import Error from "../components/Error"
import Form from "../components/Form"
import { StrictJSON } from "../lib/json"

export default function UploadView() {
    const [error, setError] = useState<string | null>(null)
    const [processing, setProcessing] = useState<boolean>(false)
    const [value, setValue] = useState<string>("")
    const [isValidValue, setIsValidValue] = useState<boolean>(false)

    function handleValue(value: string): void {
        setIsValidValue(StrictJSON.validator(value, setError))
        setValue(value)
    }

    function handleSubmit(): void {
        // Prevent incorrect JSON
        if (!isValidValue) return
        // Prevent re-submission
        if (processing) return

        setProcessing(true)
        sendRaw(value)

        // Reset
        setValue("")
        setError(null)
        setProcessing(false)
    }

    return (
        <div className="flex flex-col space-y-2">
            {error && <Error message={error} />}
            <Form onSubmit={handleSubmit}>
                <JSONUploader
                    errorCallback={setError}
                    onChange={handleValue}
                />
                {(!processing && isValidValue) && <Input
                    type="submit"
                    value="Submit"
                    className="cursor-pointer"
                />}
                {processing && <Loading color={HexColors.WHITE} />}
            </Form>
        </div>
    )
}
