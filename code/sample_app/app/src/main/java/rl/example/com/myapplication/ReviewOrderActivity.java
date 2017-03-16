package rl.example.com.myapplication;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.CheckBox;
import android.widget.Toast;

public class ReviewOrderActivity extends AppCompatActivity {

    public Toast lastToast;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_review_order);

        final CheckBox acceptTermsBox = (CheckBox) findViewById(R.id.accept_terms);
        findViewById(R.id.order_now).setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                if (!acceptTermsBox.isChecked()) {
                    lastToast = Toast.makeText(ReviewOrderActivity.this, "You must accept the terms", Toast.LENGTH_SHORT);
                    lastToast.show();
                    return;
                }
                if (lastToast != null) {
                    lastToast.cancel();
                }
                startActivity(new Intent(ReviewOrderActivity.this, OrderCompleteActivity.class));
            }
        });
    }
}
